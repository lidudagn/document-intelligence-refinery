"""
Audit Agent — multi-signal claim verification.
Optional, on-demand only (run_by_default: false).
"""
import json
import logging
import os
import re
from typing import List, Optional

from src.models.provenance import Citation
from src.models.audit_result import AuditResult, AuditVerdict
from src.agents.vector_store import VectorStoreClient
from src.agents.fact_table import FactTableDB, parse_numeric
from src.models.page_index import PageIndex
from src.utils.cache import LLMCache

logger = logging.getLogger(__name__)

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class AuditAgent:
    """
    Verifies a claim against the document corpus using triple-source retrieval:
      1. Vector search (semantic similarity)
      2. Fact table SQL (exact numeric match)
      3. PageIndex traversal (section narrowing)

    Returns an AuditResult with verdict, confidence, and evidence.
    """

    def __init__(
        self,
        vector_store: VectorStoreClient,
        fact_db: Optional[FactTableDB] = None,
        config: dict = None,
    ):
        self.vector_store = vector_store
        self.fact_db = fact_db
        self.full_config = config or {}
        self.audit_cfg = self.full_config.get("audit", {})
        self.model = os.getenv("INDEXER_LLM_MODEL", "gemini/gemini-1.5-flash")
        self.min_confidence = self.audit_cfg.get("min_verification_confidence", 0.4)

        cache_cfg = self.full_config.get("caching", {})
        self.cache = LLMCache(
            cache_dir=cache_cfg.get("cache_dir", ".refinery/cache"),
            enabled=cache_cfg.get("enabled", True),
        )

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from a claim string."""
        # Match patterns like $4.2B, 12.5%, 1,234,567
        raw_nums = re.findall(r"[\$€£]?[\d,]+\.?\d*[BMKbmk%]?", text)
        results = []
        for r in raw_nums:
            val = parse_numeric(r)
            if val is not None:
                results.append(val)
        return results

    def _vector_evidence(self, claim: str, document_id: str) -> List[dict]:
        """Retrieve chunks semantically similar to the claim."""
        if not self.vector_store.collection:
            return []
        try:
            results = self.vector_store.collection.query(
                query_texts=[claim],
                n_results=5,
                where={"document_id": document_id},
            )
            if results and results["documents"] and results["documents"][0]:
                return [
                    {"content": doc, "metadata": meta}
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0])
                ]
        except Exception as e:
            logger.warning(f"Audit vector search failed: {e}")
        return []

    def _sql_evidence(self, claim: str) -> List[dict]:
        """Search fact table for matching facts."""
        if not self.fact_db:
            return []

        results = []
        # Search by key terms
        terms = re.findall(r"\w+", claim.lower())
        stops = {"the", "was", "is", "in", "of", "a", "an", "that", "for", "to"}
        search_terms = [t for t in terms if t not in stops and len(t) > 2]

        for term in search_terms[:3]:
            try:
                hits = self.fact_db.search_facts(term)
                results.extend(hits)
            except Exception as e:
                logger.warning(f"Audit SQL search failed: {e}")

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            fid = r.get("id", "")
            if fid not in seen:
                seen.add(fid)
                unique.append(r)
        return unique[:10]

    def verify(
        self, claim: str, document_id: str, document_name: str = ""
    ) -> AuditResult:
        """Run multi-signal claim verification."""
        doc_name = document_name or document_id

        # Gather evidence from all sources
        vector_hits = self._vector_evidence(claim, document_id)
        sql_hits = self._sql_evidence(claim)

        # Count sources with evidence
        sources_with_evidence = 0
        if vector_hits:
            sources_with_evidence += 1
        if sql_hits:
            sources_with_evidence += 1

        # If no evidence at all
        if sources_with_evidence == 0:
            return AuditResult(
                claim=claim,
                verdict=AuditVerdict.UNVERIFIABLE,
                confidence=0.0,
                supporting_citations=[],
                contradicting_evidence=[],
                reasoning="No evidence found in the document corpus for this claim.",
            )

        # Build evidence text for LLM verification
        evidence_text = ""
        if vector_hits:
            evidence_text += "=== Semantic Search Results ===\n"
            for i, h in enumerate(vector_hits[:3]):
                meta = h.get("metadata", {})
                evidence_text += f"[Chunk {i+1}, page {meta.get('page_refs', '?')}]: {h['content'][:500]}\n\n"

        if sql_hits:
            evidence_text += "=== Fact Table Results ===\n"
            for f in sql_hits[:5]:
                evidence_text += f"- {f.get('metric', '?')}: {f.get('value', '?')} ({f.get('entity', '?')}, {f.get('period', '?')})\n"

        # LLM verification
        if LITELLM_AVAILABLE:
            verdict_data = self._llm_verify(claim, evidence_text)
        else:
            # Heuristic fallback: check if any SQL fact matches
            verdict_data = self._heuristic_verify(claim, sql_hits, vector_hits)

        # Build citations
        supporting = []
        for h in vector_hits[:3]:
            meta = h.get("metadata", {})
            page_str = meta.get("page_refs", "0")
            page_num = int(page_str.split(",")[0]) if page_str else 0
            supporting.append(Citation(
                document_id=document_id,
                document_name=doc_name,
                page_number=page_num,
                content_hash=meta.get("content_hash", ""),
                extracted_text=h["content"][:300],
            ))

        # Confidence based on source agreement
        confidence_map = {0: 0.0, 1: 0.4, 2: 0.7, 3: 0.9}
        confidence = confidence_map.get(sources_with_evidence, 0.4)

        verdict = AuditVerdict(verdict_data.get("verdict", "unverifiable"))

        return AuditResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            supporting_citations=supporting,
            contradicting_evidence=[],
            reasoning=verdict_data.get("reasoning", ""),
        )

    def _llm_verify(self, claim: str, evidence: str) -> dict:
        """LLM-based verification."""
        prompt = f"""Does the following evidence SUPPORT, PARTIALLY SUPPORT, CONTRADICT, or provide NO INFORMATION about this claim?

Claim: "{claim}"

Evidence:
{evidence[:4000]}

Return ONLY valid JSON:
{{"verdict": "verified|partially_verified|contradicted|unverifiable", "reasoning": "brief explanation"}}"""

        cached = self.cache.get(self.model, prompt)
        if cached:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, AttributeError):
                pass

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=15,
            )
            content = response.choices[0].message.content
            self.cache.put(self.model, prompt, content)
            return json.loads(content)
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return {"verdict": "unverifiable", "reasoning": f"LLM verification failed: {e}"}

    def _heuristic_verify(self, claim: str, sql_hits: list, vector_hits: list) -> dict:
        """Fallback verification without LLM."""
        claim_numbers = self._extract_numbers(claim)

        if sql_hits:
            for fact in sql_hits:
                fact_val = fact.get("numeric_value")
                if fact_val is not None and claim_numbers:
                    for cn in claim_numbers:
                        if abs(cn - fact_val) / max(abs(fact_val), 1) < 0.01:
                            return {"verdict": AuditVerdict.VERIFIED.value, "reasoning": f"Fact table match: {fact.get('metric')} = {fact.get('value')}"}
                        elif abs(cn - fact_val) / max(abs(fact_val), 1) < 0.20:
                            return {"verdict": AuditVerdict.PARTIALLY_VERIFIED.value, "reasoning": f"Close match: claim={cn}, fact={fact_val}"}

        if vector_hits:
            return {"verdict": AuditVerdict.PARTIALLY_VERIFIED.value, "reasoning": "Semantic matches found but no exact numeric verification."}

        return {"verdict": AuditVerdict.UNVERIFIABLE.value, "reasoning": "No matching evidence found."}
