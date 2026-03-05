import logging
from typing import List, Dict, Any, Optional
import os
import json

from src.models.page_index import PageIndex
from src.agents.vector_store import VectorStoreClient
from src.models.ldu import LDU

logger = logging.getLogger(__name__)

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class QueryAgent:
    """
    Handles user queries over a document.
    Implements a two-phase retrieval architecture:
    1. Top-Down Traversal: Uses the LLM to analyze the PageIndex tree and select the top-3 most relevant sections.
    2. Focused Dense Retrieval: Queries the vector store, heavily weighting chunks from the selected sections.
    """
    
    def __init__(self, vector_store: VectorStoreClient, config: dict = None):
        self.vector_store = vector_store
        self.config = config or {}
        self.model = os.getenv("INDEXER_LLM_MODEL", "gemini/gemini-1.5-flash")
        
    def _traverse_page_index(self, query: str, index: PageIndex) -> List[str]:
        """
        Phase 1: Ask the LLM to select the most relevant section titles
        based on the query and the section summaries/metadata.
        """
        if not LITELLM_AVAILABLE or not index.root_sections:
            return []
            
        # Build an abstract representing the document tree
        tree_text = ""
        for sec in index.root_sections:
            tree_text += f"- Title: {sec.title}\n"
            tree_text += f"  Summary: {sec.summary}\n"
            tree_text += f"  Entities: {', '.join(sec.key_entities)}\n"
            tree_text += f"  Data Types: {', '.join(sec.data_types_present)}\n\n"
            
        prompt = f"""
        You are a routing agent for a document retrieval system.
        The user asks: "{query}"
        
        Given the following hierarchical index of the document, return the exact titles 
        of up to 3 sections most likely to contain the answer.
        
        Document Index:
        {tree_text}
        
        Return ONLY valid JSON matching this schema:
        {{
            "relevant_sections": ["Section Title 1", "Section Title 2"]
        }}
        """
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=10
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("relevant_sections", [])
        except Exception as e:
            logger.warning(f"PageIndex traversal failed: {e}. Falling back to pure dense retrieval.")
            return []

    def query(self, query_text: str, document_id: str, page_index: PageIndex, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute the end-to-end RAG retrieval pipeline for a given document.
        """
        if not self.vector_store.collection:
            logger.warning("VectorStore collection unavailable. Cannot execute query.")
            return []
            
        # 1. Tree Traversal
        priority_sections = self._traverse_page_index(query_text, page_index)
        logger.info(f"LLM Traversal prioritized sections: {priority_sections}")
        
        # 2. Vector Retrieval
        # We query the ChromaDB collection. 
        # Note: In a true production system, you might issue separate filtered queries for the priority 
        # sections and combine them with a general query. Here we use a generic query but you could 
        # add 'where' clauses.
        
        # For simplicity in this implementation phase, we do a flat vector search, 
        # but filter it to only the specific document ID.
        where_filter = {"document_id": document_id}
        
        results = self.vector_store.collection.query(
            query_texts=[query_text],
            n_results=top_k * 2, # Fetch more, then re-rank/filter
            where=where_filter
        )
        
        if not results or not results['documents'] or not results['documents'][0]:
            return []
            
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results and results['distances'] else [0]*len(docs)
        
        # 3. Post-Retrieval Re-ranking (give a boost to priority sections picked by the tree traversal)
        final_results = []
        for doc_text, meta, dist in zip(docs, metas, distances):
            score = float(1.0 / (1.0 + dist)) # inverse distance
            
            # Boost if the chunk belongs to a section the LLM indexer thought was relevant
            section_name = meta.get("parent_section", "")
            if priority_sections and section_name in priority_sections:
                score *= 1.5 # 50% boost
                
            final_results.append({
                "content": doc_text,
                "metadata": meta,
                "score": score
            })
            
        # Sort by boosted score
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]
