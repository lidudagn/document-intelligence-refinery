"""
LangGraph Query Agent — 3-tool architecture with deterministic routing,
hybrid retrieval (BM25 + vector), citation validation, and SQL-direct shortcut.
"""
import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, TypedDict

from src.models.page_index import PageIndex
from src.models.provenance import ProvenanceChain, Citation
from src.agents.vector_store import VectorStoreClient
from src.agents.fact_table import FactTableDB
from src.models.ldu import LDU, ChunkType
from src.utils.cache import LLMCache
from src.utils.metrics import QueryMetrics

logger = logging.getLogger(__name__)

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class QueryState(TypedDict, total=False):
    query: str
    document_id: str
    document_name: str
    page_index: dict
    priority_sections: list[str]
    retrieved_chunks: list[dict]
    fact_results: list[dict]
    answer: str
    provenance_chain: dict
    tool_calls_made: list[str]
    route: str  # which tools the orchestrator decided to run


# ---------------------------------------------------------------------------
# Deterministic Orchestrator (NO LLM)
# ---------------------------------------------------------------------------

_NUMERIC_PATTERNS = re.compile(
    r"(how much|how many|total|sum|amount|revenue|income|expense|cost|"
    r"profit|loss|tax|budget|capital|expenditure|billion|million|"
    r"\$|etb|birr|\d+[.,]\d+|\d{4,})",
    re.IGNORECASE,
)

_SECTION_KEYWORDS = re.compile(
    r"(section|chapter|part|appendix|annex|introduction|conclusion|"
    r"findings|recommendation|methodology|overview|summary)",
    re.IGNORECASE,
)


def deterministic_route(query: str) -> str:
    """Pattern-match router — zero LLM cost."""
    has_numeric = bool(_NUMERIC_PATTERNS.search(query))
    has_section = bool(_SECTION_KEYWORDS.search(query))

    if has_numeric:
        return "sql_then_hybrid"
    elif has_section:
        return "navigate_then_hybrid"
    else:
        return "hybrid_only"


# ---------------------------------------------------------------------------
# BM25-lite keyword search (no external dependency)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25Lite:
    """Minimal BM25 implementation for keyword retrieval over LDU content."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[dict] = []
        self._doc_tokens: List[List[str]] = []
        self._avg_dl: float = 0.0
        self._df: Counter = Counter()
        self._n: int = 0

    def index(self, docs: List[dict]) -> None:
        """Index documents (list of dicts with 'content' key)."""
        self._docs = docs
        self._doc_tokens = [_tokenize(d.get("content", "")) for d in docs]
        self._n = len(docs)
        self._avg_dl = sum(len(t) for t in self._doc_tokens) / max(1, self._n)
        self._df = Counter()
        for tokens in self._doc_tokens:
            for t in set(tokens):
                self._df[t] += 1

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        query_tokens = _tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self._doc_tokens):
            score = 0.0
            dl = len(doc_tokens)
            tf_map = Counter(doc_tokens)
            for qt in query_tokens:
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                df = self._df.get(qt, 0)
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                score += idf * (numerator / denominator)
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score <= 0:
                break
            doc = dict(self._docs[idx])
            doc["bm25_score"] = score
            results.append(doc)
        return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    vector_results: List[dict],
    keyword_results: List[dict],
    k: int = 60,
) -> List[dict]:
    """Merge vector and keyword results using RRF."""
    # Track by content_hash for dedup
    scores: Dict[str, float] = {}
    docs: Dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        key = doc.get("metadata", {}).get("content_hash", str(rank))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        docs[key] = doc

    for rank, doc in enumerate(keyword_results):
        key = doc.get("content_hash", doc.get("metadata", {}).get("content_hash", f"bm25_{rank}"))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in docs:
            # Normalize BM25 result to same shape as vector result
            docs[key] = {
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "score": doc.get("bm25_score", 0.0),
            }

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"fused_score": s, **docs[key]} for key, s in ranked]


# ---------------------------------------------------------------------------
# Query Agent
# ---------------------------------------------------------------------------

class QueryAgent:
    """
    LangGraph-orchestrated query agent with 3 tools:
    1. pageindex_navigate  — LLM tree traversal
    2. hybrid_search       — BM25 + ChromaDB vector + section boost
    3. structured_query    — SQL over FactTable

    Deterministic routing. SQL-direct-answer shortcut.
    Citation validation. Retrieval confidence gate.
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
        self.config = self.full_config.get("query_agent", {})
        self.model = self.config.get("llm_model", os.getenv("INDEXER_LLM_MODEL", "gemini/gemini-1.5-flash"))
        self.top_k = self.config.get("top_k_retrieval", 10)
        self.max_chunks = self.config.get("max_chunks_for_synthesis", 6)
        self.min_confidence = self.config.get("min_retrieval_confidence", 0.3)
        self.section_boost = self.config.get("section_boost_factor", 1.5)
        self.sql_direct = self.config.get("sql_direct_answer", True)
        self.max_tokens = self.config.get("max_synthesis_tokens", 6000)

        cache_cfg = self.full_config.get("caching", {})
        self.cache = LLMCache(
            cache_dir=cache_cfg.get("cache_dir", ".refinery/cache"),
            enabled=cache_cfg.get("enabled", True),
        )
        self.metrics = QueryMetrics()
        self.bm25 = BM25Lite()
        self._bm25_indexed = False

    # -----------------------------------------------------------------------
    # Tool 1: PageIndex Navigate
    # -----------------------------------------------------------------------

    def _pageindex_navigate(self, query: str, page_index: PageIndex) -> List[str]:
        """LLM traversal to find relevant sections."""
        if not LITELLM_AVAILABLE or not page_index.root_sections:
            return []

        tree_text = ""
        for sec in page_index.root_sections:
            tree_text += f"- {sec.title} (pages {sec.page_start}-{sec.page_end})\n"
            tree_text += f"  Summary: {sec.summary}\n"
            tree_text += f"  Entities: {', '.join(sec.key_entities)}\n\n"

        prompt = f"""Given this document index, return the titles of up to 3 sections most likely to answer: "{query}"

Document Index:
{tree_text}

Return ONLY valid JSON: {{"relevant_sections": ["Title 1", "Title 2"]}}"""

        cached = self.cache.get(self.model, prompt)
        if cached:
            try:
                return json.loads(cached).get("relevant_sections", [])
            except (json.JSONDecodeError, AttributeError):
                pass

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=10,
            )
            content = response.choices[0].message.content
            self.cache.put(self.model, prompt, content)
            return json.loads(content).get("relevant_sections", [])
        except Exception as e:
            logger.warning(f"PageIndex traversal failed: {e}")
            return []

    # -----------------------------------------------------------------------
    # Tool 2: Hybrid Search (BM25 + Vector + Section Boost)
    # -----------------------------------------------------------------------

    def _ensure_bm25_indexed(self, document_id: str) -> None:
        """Index BM25 from ChromaDB stored docs (lazy, once per query session)."""
        if self._bm25_indexed or not self.vector_store.collection:
            return
        try:
            all_docs = self.vector_store.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )
            if all_docs and all_docs["documents"]:
                docs_for_bm25 = []
                for doc_text, meta in zip(all_docs["documents"], all_docs["metadatas"]):
                    docs_for_bm25.append({"content": doc_text, "metadata": meta})
                self.bm25.index(docs_for_bm25)
                self._bm25_indexed = True
        except Exception as e:
            logger.warning(f"BM25 indexing failed: {e}")

    def _hybrid_search(
        self, query: str, document_id: str, priority_sections: List[str]
    ) -> List[dict]:
        """BM25 + ChromaDB vector → RRF merge → section boost → top-k."""
        vector_results = []
        keyword_results = []

        # Vector search
        if self.vector_store.collection:
            try:
                raw = self.vector_store.collection.query(
                    query_texts=[query],
                    n_results=self.top_k,
                    where={"document_id": document_id},
                )
                if raw and raw["documents"] and raw["documents"][0]:
                    dists = raw.get("distances", [[]])[0]
                    for i, (doc, meta) in enumerate(
                        zip(raw["documents"][0], raw["metadatas"][0])
                    ):
                        dist = dists[i] if i < len(dists) else 0
                        vector_results.append({
                            "content": doc,
                            "metadata": meta,
                            "score": 1.0 / (1.0 + dist),
                        })
            except Exception as e:
                logger.warning(f"Vector search failed: {e}. Falling back to keyword only.")

        # Keyword search
        self._ensure_bm25_indexed(document_id)
        if self._bm25_indexed:
            keyword_results = self.bm25.search(query, top_k=self.top_k)

        # Fuse
        if vector_results and keyword_results:
            merged = reciprocal_rank_fusion(vector_results, keyword_results)
        elif vector_results:
            merged = vector_results
        elif keyword_results:
            merged = [{"content": r["content"], "metadata": r.get("metadata", {}),
                        "score": r.get("bm25_score", 0), "fused_score": r.get("bm25_score", 0)}
                       for r in keyword_results]
        else:
            return []

        # Section boost
        for item in merged:
            section = item.get("metadata", {}).get("parent_section", "")
            if priority_sections and section in priority_sections:
                item["score"] = item.get("fused_score", item.get("score", 0)) * self.section_boost
            else:
                item["score"] = item.get("fused_score", item.get("score", 0))

        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:self.top_k]

    # -----------------------------------------------------------------------
    # Tool 3: Structured Query (SQL over FactTable)
    # -----------------------------------------------------------------------

    def _structured_query(self, query: str) -> List[dict]:
        """Translate query → SQL keyword search over fact table."""
        if not self.fact_db:
            return []

        # Extract key terms for fuzzy search
        terms = re.findall(r"\w+", query.lower())
        # Remove stopwords
        stops = {"what", "is", "the", "of", "in", "for", "a", "an", "was", "were", "how", "much", "many", "total"}
        search_terms = [t for t in terms if t not in stops and len(t) > 2]

        all_results = []
        for term in search_terms[:5]:  # limit queries
            try:
                results = self.fact_db.search_facts(term)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"SQL search failed for '{term}': {e}")

        # Deduplicate by fact id
        seen = set()
        unique = []
        for r in all_results:
            fid = r.get("id", "")
            if fid not in seen:
                seen.add(fid)
                unique.append(r)

        return unique[:10]

    # -----------------------------------------------------------------------
    # Synthesizer
    # -----------------------------------------------------------------------

    def _compute_evidence_hash(self, query: str, chunks: List[dict]) -> str:
        hashes = sorted(
            c.get("metadata", {}).get("content_hash", "") for c in chunks
        )
        raw = f"{query}::{'|'.join(hashes)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _validate_citations(self, citations: List[Citation], chunks: List[dict]) -> List[Citation]:
        """Post-synthesis guard: verify each citation's text exists in retrieved chunks."""
        chunk_texts = {c.get("content", "")[:200] for c in chunks}
        valid = []
        for cit in citations:
            # Check if the first 100 chars of the cited text appear in any chunk
            snippet = cit.extracted_text[:100]
            if any(snippet[:50] in ct for ct in chunk_texts):
                valid.append(cit)
            else:
                logger.warning(f"Citation validation failed: '{snippet[:50]}...' not found in chunks")
        return valid

    def _build_citations_from_chunks(self, chunks: List[dict], document_name: str) -> List[Citation]:
        """Build Citation objects from retrieved chunks."""
        citations = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            page_refs_str = meta.get("page_refs", "0")
            page_num = int(page_refs_str.split(",")[0]) if page_refs_str else 0

            bbox = None
            if "bbox" in meta and meta["bbox"]:
                try:
                    bbox = [float(x) for x in meta["bbox"].split(",")]
                except (ValueError, AttributeError):
                    bbox = None

            citations.append(Citation(
                document_id=meta.get("document_id", ""),
                document_name=document_name,
                page_number=page_num,
                bbox=bbox,
                content_hash=meta.get("content_hash", ""),
                extracted_text=chunk.get("content", "")[:500],
            ))
        return citations

    def _synthesize_with_llm(
        self, query: str, chunks: List[dict], fact_results: List[dict], document_name: str
    ) -> ProvenanceChain:
        """LLM synthesis with full provenance."""
        # Truncate chunks to max_chunks
        top_chunks = chunks[:self.max_chunks]

        # Build evidence text
        evidence_parts = []
        for i, c in enumerate(top_chunks):
            meta = c.get("metadata", {})
            evidence_parts.append(
                f"[Source {i+1}, page {meta.get('page_refs', '?')}, section: {meta.get('parent_section', '?')}]\n"
                f"{c.get('content', '')[:1000]}"
            )

        # Add fact results
        for f in fact_results[:5]:
            evidence_parts.append(
                f"[SQL Fact: {f.get('metric', '?')} = {f.get('value', '?')} "
                f"({f.get('entity', '?')}, {f.get('period', '?')}), page {f.get('page_number', '?')}]"
            )

        evidence_text = "\n\n".join(evidence_parts)

        # Token budget check
        if len(evidence_text) > self.max_tokens * 4:  # rough char-to-token ratio
            evidence_text = evidence_text[:self.max_tokens * 4]

        prompt = f"""Based ONLY on the following evidence, answer the query.
For every claim, cite the source [Source N, page X].
If the evidence is insufficient, say "Information not found in the document."

Query: {query}

Evidence:
{evidence_text}

Answer:"""

        citations = self._build_citations_from_chunks(top_chunks, document_name)
        evidence_hash = self._compute_evidence_hash(query, top_chunks)
        retrieval_methods = ["hybrid_search"]
        if fact_results:
            retrieval_methods.append("structured_query")

        # Check cache
        cached = self.cache.get(self.model, prompt)
        if cached:
            answer_text = cached
        elif LITELLM_AVAILABLE:
            try:
                response = completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30,
                )
                answer_text = response.choices[0].message.content
                self.cache.put(self.model, prompt, answer_text)
                # Track tokens
                usage = getattr(response, "usage", None)
                if usage:
                    self.metrics.record_tokens(
                        usage.total_tokens,
                        cost_usd=getattr(usage, "total_cost", 0.0) or 0.0,
                    )
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
                answer_text = "Synthesis failed. Retrieved evidence is available but could not be summarized."
        else:
            answer_text = "LLM not available. Raw evidence retrieved."

        # Validate citations
        valid_citations = self._validate_citations(citations, top_chunks)

        scores = [c.get("score", 0) for c in top_chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return ProvenanceChain(
            answer=answer_text,
            query_text=query,
            citations=valid_citations,
            is_verifiable=len(valid_citations) > 0,
            confidence_level=round(min(avg_score, 1.0), 3),
            retrieval_method="+".join(retrieval_methods),
            evidence_bundle_hash=evidence_hash,
        )

    # -----------------------------------------------------------------------
    # Main query entry point
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        document_id: str,
        page_index: PageIndex,
        document_name: str = "",
    ) -> ProvenanceChain:
        """Execute the full query pipeline."""
        self.metrics.start(query_text, document_id)
        doc_name = document_name or document_id

        # 1. Deterministic routing
        route = deterministic_route(query_text)
        logger.info(f"Query route: {route}")

        priority_sections: List[str] = []
        fact_results: List[dict] = []
        retrieved_chunks: List[dict] = []

        # 2. Execute tools based on route
        if route == "sql_then_hybrid":
            # Try SQL first
            self.metrics.record_tool("structured_query")
            fact_results = self._structured_query(query_text)

            # SQL-direct-answer shortcut
            if fact_results and self.sql_direct and len(fact_results) <= 3:
                self.metrics.record_sql_direct()
                answer_parts = []
                citations = []
                for f in fact_results:
                    answer_parts.append(f"{f.get('metric', '?')}: {f.get('value', '?')} ({f.get('period', '?')})")
                    citations.append(Citation(
                        document_id=document_id,
                        document_name=doc_name,
                        page_number=f.get("page_number", 0),
                        bbox=[float(x) for x in json.loads(f["bbox"])] if f.get("bbox") else None,
                        content_hash=f.get("content_hash", ""),
                        extracted_text=f"{f.get('metric', '')}: {f.get('value', '')}",
                    ))

                chain = ProvenanceChain(
                    answer="; ".join(answer_parts),
                    query_text=query_text,
                    citations=citations,
                    is_verifiable=True,
                    confidence_level=0.95,
                    retrieval_method="structured_query_direct",
                    evidence_bundle_hash=self._compute_evidence_hash(query_text, []),
                )
                self.metrics.finish_and_log()
                return chain

            # Fall through to hybrid for context
            self.metrics.record_tool("hybrid_search")
            priority_sections = self._pageindex_navigate(query_text, page_index)
            self.metrics.record_tool("pageindex_navigate")
            retrieved_chunks = self._hybrid_search(query_text, document_id, priority_sections)

        elif route == "navigate_then_hybrid":
            self.metrics.record_tool("pageindex_navigate")
            priority_sections = self._pageindex_navigate(query_text, page_index)
            self.metrics.record_tool("hybrid_search")
            retrieved_chunks = self._hybrid_search(query_text, document_id, priority_sections)

        else:  # hybrid_only
            self.metrics.record_tool("hybrid_search")
            retrieved_chunks = self._hybrid_search(query_text, document_id, [])

        # Record retrieval scores
        scores = [c.get("score", 0) for c in retrieved_chunks]
        self.metrics.record_retrieval_scores(scores)

        # 3. Retrieval confidence gate
        top_score = max(scores) if scores else 0.0
        if top_score < self.min_confidence:
            logger.info(f"Retrieval confidence {top_score:.3f} < {self.min_confidence}. Marking unverifiable.")
            chain = ProvenanceChain(
                answer="Information not found in the document with sufficient confidence.",
                query_text=query_text,
                citations=[],
                is_verifiable=False,
                confidence_level=round(top_score, 3),
                retrieval_method="hybrid_search",
                evidence_bundle_hash=self._compute_evidence_hash(query_text, retrieved_chunks),
            )
            self.metrics.finish_and_log()
            return chain

        # 4. Synthesize
        chain = self._synthesize_with_llm(query_text, retrieved_chunks, fact_results, doc_name)
        self.metrics.finish_and_log()
        return chain
