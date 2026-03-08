"""
FactTable Extractor — extracts key-value facts from table LDUs into a
per-document SQLite database.  Heuristic-first with LLM fallback.
"""
import hashlib
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.models.ldu import LDU, ChunkType
from src.models.fact_record import FactRecord
from src.utils.cache import LLMCache

logger = logging.getLogger(__name__)

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

# Symbols to strip before attempting float conversion
_STRIP_SYMBOLS = re.compile(r"[$€£,\s]|(?i:ETB)")
_PERCENTAGE = re.compile(r"^([\d.]+)\s*%$")
_PAREN_NEGATIVE = re.compile(r"^\(([\d,.]+)\)$")  # accounting negative: (1,234)


def parse_numeric(raw: str) -> Optional[float]:
    """Best-effort parse a value string into a float."""
    s = raw.strip()
    if not s or s == "-" or s.lower() in ("n/a", "nil", "—", "–"):
        return None

    # Percentage
    m = _PERCENTAGE.match(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Accounting negative: (1,234.56)
    m = _PAREN_NEGATIVE.match(s)
    if m:
        cleaned = _STRIP_SYMBOLS.sub("", m.group(1))
        try:
            return -float(cleaned)
        except ValueError:
            pass

    cleaned = _STRIP_SYMBOLS.sub("", s)
    multiplier = 1.0
    if cleaned.lower().endswith("b"):
        multiplier = 1e9
        cleaned = cleaned[:-1]
    elif cleaned.lower().endswith("m"):
        multiplier = 1e6
        cleaned = cleaned[:-1]
    elif cleaned.lower().endswith("k"):
        multiplier = 1e3
        cleaned = cleaned[:-1]

    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def _detect_unit(raw: str) -> str:
    s = raw.strip().lower()
    if "$" in raw:
        return "USD"
    if "etb" in s or "birr" in s:
        return "ETB"
    if "€" in raw:
        return "EUR"
    if "£" in raw:
        return "GBP"
    if "%" in raw:
        return "percentage"
    return ""


# ---------------------------------------------------------------------------
# Period / Entity heuristics
# ---------------------------------------------------------------------------

_PERIOD_PATTERNS = [
    re.compile(r"(FY\s*\d{4}(?:[/-]\d{2,4})?)", re.IGNORECASE),
    re.compile(r"(Q[1-4]\s*\d{4})", re.IGNORECASE),
    re.compile(r"(\d{4}[/-]\d{2,4})"),
    re.compile(r"(\d{4})"),
]


def _detect_period(text: str) -> Optional[str]:
    for pat in _PERIOD_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    entity TEXT,
    period TEXT,
    value TEXT NOT NULL,
    unit TEXT,
    numeric_value REAL,
    page_number INTEGER,
    bbox TEXT,
    content_hash TEXT,
    section TEXT,
    confidence REAL
);
"""

_READ_ONLY_GUARD = re.compile(
    r"^\s*SELECT\b", re.IGNORECASE
)

_DANGEROUS_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)


class FactTableDB:
    """Thin SQLite wrapper — one .db file per document."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(_CREATE_TABLE_SQL)
        self.conn.commit()

    def insert_facts(self, facts: List[FactRecord]) -> int:
        sql = """
        INSERT OR REPLACE INTO facts
        (id, doc_id, metric, entity, period, value, unit, numeric_value,
         page_number, bbox, content_hash, section, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = []
        for f in facts:
            bbox_str = json.dumps(f.bbox) if f.bbox else None
            rows.append((
                f.fact_id, f.document_id, f.metric, f.entity, f.period,
                f.value, f.unit, f.numeric_value, f.page_number, bbox_str,
                f.content_hash, f.section, f.confidence,
            ))
        self.conn.executemany(sql, rows)
        self.conn.commit()
        return len(rows)

    def query_facts(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a read-only SELECT query. Rejects anything else."""
        if not _READ_ONLY_GUARD.match(sql):
            raise ValueError("Only SELECT queries are allowed.")
        if _DANGEROUS_KEYWORDS.search(sql):
            raise ValueError(f"Dangerous SQL keyword detected in query: {sql[:80]}")
        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def search_facts(self, keyword: str) -> List[Dict[str, Any]]:
        """Fuzzy search across metric, entity, and value columns."""
        sql = "SELECT * FROM facts WHERE metric LIKE ? OR entity LIKE ? OR value LIKE ?"
        pattern = f"%{keyword}%"
        return self.query_facts(sql, (pattern, pattern, pattern))

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class FactTableExtractor:
    """
    Extracts structured facts from table-type LDUs.
    Strategy: heuristic-first, LLM fallback.
    """

    def __init__(self, config: dict = None):
        self.config = (config or {}).get("fact_table", {})
        self.mode = self.config.get("extraction_mode", "heuristic_first")
        self.model = os.getenv("INDEXER_LLM_MODEL", "gemini/gemini-2.0-flash")
        cache_cfg = (config or {}).get("caching", {})
        self.cache = LLMCache(
            cache_dir=cache_cfg.get("cache_dir", ".refinery/cache"),
            enabled=cache_cfg.get("enabled", True),
        )

    def extract_from_ldus(
        self, document_id: str, ldus: List[LDU], db_path: Optional[str] = None
    ) -> List[FactRecord]:
        """Extract facts from all table LDUs and persist to SQLite."""
        table_ldus = [l for l in ldus if l.chunk_type == ChunkType.TABLE]
        if not table_ldus:
            logger.info(f"No table LDUs found for {document_id}. Skipping fact extraction.")
            return []

        all_facts: List[FactRecord] = []
        for ldu in table_ldus:
            facts = self._extract_single_table(document_id, ldu)
            all_facts.extend(facts)

        if not all_facts:
            return []

        # Persist to SQLite
        if db_path is None:
            db_path = f".refinery/fact_tables/{document_id}.db"
        db = FactTableDB(db_path)
        count = db.insert_facts(all_facts)
        db.close()
        logger.info(f"Extracted {count} facts from {len(table_ldus)} tables → {db_path}")
        return all_facts

    def _extract_single_table(self, document_id: str, ldu: LDU) -> List[FactRecord]:
        """Try heuristic first, fall back to LLM if needed."""
        meta = ldu.metadata or {}
        headers = meta.get("raw_headers", [])
        rows = meta.get("raw_rows", [])

        facts: List[FactRecord] = []

        # Path 1: Heuristic extraction (free)
        if headers and rows and self.mode != "llm_only":
            facts = self._heuristic_extract(document_id, ldu, headers, rows)
            if facts or self.mode == "heuristic_only":
                return facts

        # Path 2: LLM fallback
        if self.mode != "heuristic_only" and LITELLM_AVAILABLE:
            facts = self._llm_extract(document_id, ldu)

        return facts

    def _heuristic_extract(
        self, document_id: str, ldu: LDU,
        headers: List[str], rows: List[List[str]]
    ) -> List[FactRecord]:
        """Parse table rows using column-header alignment."""
        facts = []
        # The first column is typically the metric/category label
        # Remaining columns are values (possibly for different periods)
        if len(headers) < 2:
            return []

        metric_col = 0
        value_cols = list(range(1, len(headers)))

        for row in rows:
            if len(row) <= metric_col:
                continue
            metric_name = str(row[metric_col]).strip()
            if not metric_name or metric_name == "-":
                continue

            for vi in value_cols:
                if vi >= len(row):
                    continue
                raw_value = str(row[vi]).strip()
                if not raw_value or raw_value == "-":
                    continue

                header_text = headers[vi] if vi < len(headers) else ""
                detected_period = _detect_period(header_text)
                numeric = parse_numeric(raw_value)
                unit = _detect_unit(raw_value)

                fact_id = hashlib.sha256(
                    f"{document_id}:{ldu.chunk_id}:{metric_name}:{header_text}".encode()
                ).hexdigest()[:16]

                facts.append(FactRecord(
                    fact_id=fact_id,
                    document_id=document_id,
                    metric=metric_name,
                    entity=None,
                    period=detected_period,
                    value=raw_value,
                    unit=unit,
                    numeric_value=numeric,
                    page_number=ldu.page_refs[0] if ldu.page_refs else 0,
                    bbox=[ldu.bounding_box.x0, ldu.bounding_box.top,
                          ldu.bounding_box.x1, ldu.bounding_box.bottom] if ldu.bounding_box else None,
                    content_hash=ldu.content_hash,
                    section=ldu.parent_section,
                    confidence=0.85,  # heuristic confidence
                ))

        return facts

    def _llm_extract(self, document_id: str, ldu: LDU) -> List[FactRecord]:
        """Send table text to LLM for structured extraction."""
        prompt = f"""Extract key financial/numerical facts from this table.
Return a JSON array where each element has:
  "metric": the measured quantity name
  "entity": the organization (if identifiable)
  "period": the time period (if identifiable)
  "value": the raw value string
  "unit": the unit (USD, ETB, percentage, count, etc.)

Table content:
{ldu.content[:4000]}

Return ONLY valid JSON array. No other text."""

        # Check cache first
        cached = self.cache.get(self.model, prompt)
        if cached:
            raw_json = cached
        else:
            try:
                response = completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    timeout=15,
                )
                raw_json = response.choices[0].message.content
                self.cache.put(self.model, prompt, raw_json)
            except Exception as e:
                logger.warning(f"LLM fact extraction failed: {e}")
                return []

        try:
            data = json.loads(raw_json)
            # Handle both {"facts": [...]} and bare [...]
            items = data if isinstance(data, list) else data.get("facts", data.get("results", []))
        except (json.JSONDecodeError, AttributeError):
            return []

        facts = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            metric = item.get("metric", "")
            if not metric:
                continue

            raw_val = item.get("value", "")
            fact_id = hashlib.sha256(
                f"{document_id}:{ldu.chunk_id}:llm:{i}:{metric}".encode()
            ).hexdigest()[:16]

            facts.append(FactRecord(
                fact_id=fact_id,
                document_id=document_id,
                metric=metric,
                entity=item.get("entity"),
                period=item.get("period"),
                value=raw_val,
                unit=item.get("unit", _detect_unit(raw_val)),
                numeric_value=parse_numeric(raw_val),
                page_number=ldu.page_refs[0] if ldu.page_refs else 0,
                bbox=[ldu.bounding_box.x0, ldu.bounding_box.top,
                      ldu.bounding_box.x1, ldu.bounding_box.bottom] if ldu.bounding_box else None,
                content_hash=ldu.content_hash,
                section=ldu.parent_section,
                confidence=0.70,  # LLM confidence is lower than heuristic
            ))

        return facts
