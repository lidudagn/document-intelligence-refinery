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
        Deduplicates by content_hash before ingestion.
        """
        if not CHROMA_AVAILABLE:
            logger.info(f"Mock ingestion: skipped {len(ldus)} chunks from {document_id}")
            return
            
        if not ldus:
            return
        
        # Deduplicate by content_hash (removes repeated headers/footers)
        seen_hashes: set = set()
        unique_ldus: List[LDU] = []
        for ldu in ldus:
            if ldu.content_hash not in seen_hashes:
                seen_hashes.add(ldu.content_hash)
                unique_ldus.append(ldu)
        
        dedup_count = len(ldus) - len(unique_ldus)
        if dedup_count > 0:
            logger.info(f"Deduplication: removed {dedup_count} duplicate chunks (by content_hash)")
            
        ids = []
        documents = []
        metadatas = []
        
        for ldu in unique_ldus:
            meta: Dict[str, str | int | float | bool] = {
                "document_id": document_id,
                "chunk_type": ldu.chunk_type.value,
                "parent_section": ldu.parent_section or "unknown",
                "token_count": ldu.token_count,
                "content_hash": ldu.content_hash,
                "page_refs": ",".join(str(p) for p in ldu.page_refs)
            }
            
            if ldu.bounding_box:
                meta["bbox"] = f"{ldu.bounding_box.x0:.1f},{ldu.bounding_box.top:.1f},{ldu.bounding_box.x1:.1f},{ldu.bounding_box.bottom:.1f}"
                
            ids.append(ldu.chunk_id)
            documents.append(ldu.content)
            metadatas.append(meta)
            
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Ingested {len(ids)} LDUs into ChromaDB for {document_id}")
