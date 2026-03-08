import logging
import os
import json
from typing import List, Dict, Optional
from pydantic import BaseModel

from src.models.page_index import PageIndex, Section
from src.models.ldu import LDU, ChunkType

# Use LiteLLM if available, otherwise just use heuristic fallback
try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

logger = logging.getLogger(__name__)

class SectionSummaryResponse(BaseModel):
    summary: str
    key_entities: List[str]
    data_types_present: List[str]

class PageIndexBuilder:
    """
    Builds a hierarchical PageIndex tree from a sequence of LDUs.
    Utilizes an LLM to generate rich summaries and entities, but gracefully
    degrades to heuristic extraction if the LLM fails or is unavailable.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        # Use the same model from extraction_rules.yaml -> query_agent.llm_model
        qa_config = self.config.get("query_agent", {})
        self.model = qa_config.get("llm_model", os.getenv("INDEXER_LLM_MODEL", "openrouter/qwen/qwen-2.5-72b-instruct"))
        
    def _generate_heuristic_summary(self, text: str) -> SectionSummaryResponse:
        """Fallback: Extracts the first 2-3 sentences as a summary."""
        sentences = text.split(". ")
        summary = ". ".join(sentences[:3]) + ("." if sentences[:3] and not sentences[:3][-1].endswith(".") else "")
        
        # Simple heuristic data types
        data_types = []
        if "table" in text.lower() or "|" in text:
            data_types.append("tables")
        if "figure" in text.lower() or "chart" in text.lower():
            data_types.append("figures")
            
        return SectionSummaryResponse(
            summary=summary,
            key_entities=[],
            data_types_present=data_types
        )
        
    def _generate_llm_summary(self, text: str) -> SectionSummaryResponse:
        """Call LLM to get a structured summary of the section."""
        if not LITELLM_AVAILABLE:
            return self._generate_heuristic_summary(text)
            
        # Truncate text to avoid massive token costs for just a summary
        truncated_text = text[:8000] 
        
        prompt = f"""
        Analyze this document section and provide a JSON response containing:
        1. "summary": A 2-3 sentence concise summary.
        2. "key_entities": A list of max 5 important named entities (people, orgs, regulations, metrics).
        3. "data_types_present": A list of structural elements present (e.g. "financial_tables", "legal_clauses", "figures").
        
        Section Text:
        {truncated_text}
        
        Return ONLY valid JSON matching this schema:
        {{
            "summary": "...",
            "key_entities": ["..."],
            "data_types_present": ["..."]
        }}
        """
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=60 # Increased for OpenRouter stability
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            return SectionSummaryResponse(
                summary=data.get("summary", ""),
                key_entities=data.get("key_entities", []),
                data_types_present=data.get("data_types_present", [])
            )
            
        except Exception as e:
            logger.warning(f"LLM summarization failed ({e}). Falling back to heuristics.")
            return self._generate_heuristic_summary(text)

    def build_index(self, document_id: str, ldus: List[LDU]) -> PageIndex:
        """
        Flat to Hierarchical conversion.
        Groups LDUs by their `parent_section` and builds Section nodes.
        Note: True nested hierarchy (H1 -> H2 -> H3) requires markdown header levels (# vs ##).
        For this implementation, we build a flat list of sections based on active_parent.
        """
        
        if not ldus:
            return PageIndex(document_id=document_id, root_sections=[])
            
        # Group LDUs by section
        section_map: Dict[str, List[LDU]] = {}
        DEFAULT_SECTION = "Introduction"
        
        for ldu in ldus:
            sec_name = ldu.parent_section or DEFAULT_SECTION
            if sec_name not in section_map:
                section_map[sec_name] = []
            section_map[sec_name].append(ldu)
            
        root_sections: List[Section] = []
        
        for sec_name, sec_ldus in section_map.items():
            # Calculate page bounds
            all_pages = [p for ldu in sec_ldus for p in ldu.page_refs]
            page_start = min(all_pages) if all_pages else 0
            page_end = max(all_pages) if all_pages else 0
            
            # Aggregate text for summarization
            full_text = "\n".join(ldu.content for ldu in sec_ldus if ldu.chunk_type == ChunkType.TEXT)
            
            # Generate summary and metadata
            summary_data = self._generate_llm_summary(full_text)
            
            # Add table/figure detection directly from chunk types
            for ldu in sec_ldus:
                if ldu.chunk_type == ChunkType.TABLE and "tables" not in summary_data.data_types_present:
                    summary_data.data_types_present.append("tables")
            
            section = Section(
                title=sec_name,
                page_start=page_start,
                page_end=page_end,
                child_sections=[], # Nested sections would go here if parsed recursively
                key_entities=summary_data.key_entities,
                summary=summary_data.summary,
                data_types_present=summary_data.data_types_present
            )
            root_sections.append(section)
            
        return PageIndex(
            document_id=document_id,
            root_sections=root_sections
        )
