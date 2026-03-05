import os
import logging
from typing import List, Dict, Any

from src.models.ldu import LDU, ChunkType

logger = logging.getLogger(__name__)

# Lightweight local fallback if chromadb isn't available
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class VectorStoreClient:
    """
    Ingests LDUs into a local ChromaDB instance to enable RAG.
    """
    
    def __init__(self, db_path: str = ".refinery/chroma_db", collection_name: str = "ldu_collection"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        if CHROMA_AVAILABLE:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            logger.warning("chromadb is not installed. VectorStoreClient will run in mock mode.")
            self.client = None
            self.collection = None

    def ingest_ldus(self, document_id: str, ldus: List[LDU]):
        """
        Takes a list of validated LDUs and syncs them to ChromaDB.
        Skips ingestion if ChromaDB is not installed.
        """
        if not CHROMA_AVAILABLE:
            logger.info(f"Mock ingestion: skipped {len(ldus)} chunks from {document_id}")
            return
            
        if not ldus:
            return
            
        ids = []
        documents = []
        metadatas = []
        
        for ldu in ldus:
            # We must convert ChunkType/Lists/Dicts to basic types for ChromaDB metadata
            meta: Dict[str, str | int | float | bool] = {
                "document_id": document_id,
                "chunk_type": ldu.chunk_type.value,
                "parent_section": ldu.parent_section or "unknown",
                "token_count": ldu.token_count,
                "content_hash": ldu.content_hash,
                "page_refs": ",".join(str(p) for p in ldu.page_refs)
            }
            
            # Add bbox as string if exists
            if ldu.bounding_box:
                meta["bbox"] = f"{ldu.bounding_box.x0:.1f},{ldu.bounding_box.top:.1f},{ldu.bounding_box.x1:.1f},{ldu.bounding_box.bottom:.1f}"
                
            ids.append(ldu.chunk_id)
            documents.append(ldu.content)
            metadatas.append(meta)
            
        # Upsert in small batches if necessary, but typically a doc's ldus fit in one call
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Ingested {len(ids)} LDUs into ChromaDB for {document_id}")
