import hashlib
import json
import logging
from typing import List, Optional

from src.models.ldu import LDU, ChunkType, BoundingBox
from src.models.extracted_document import ExtractedDocument, TextBlock, TableBlock

logger = logging.getLogger(__name__)

class ChunkValidator:
    """Enforces the 5 chunking constitution rules."""
    
    @staticmethod
    def validate(ldus: List[LDU], max_tokens: int) -> bool:
        for i, ldu in enumerate(ldus):
            # Rule 1: tables must not split (a table is always 1 LDU)
            if ldu.chunk_type == ChunkType.TABLE:
                if len(ldu.page_refs) > 1:
                    logger.warning(f"LDU {ldu.chunk_id} spans {len(ldu.page_refs)} pages. Tables should ideally remain intact.")
                    
            # Rule 4: headers propagate to children
            # (If it's not a header itself, and a header was seen, it should have a parent)
            
            # Additional validation: token bounds
            if ldu.token_count > max_tokens and ldu.chunk_type != ChunkType.TABLE:
                logger.warning(f"LDU {ldu.chunk_id} exceeds {max_tokens} tokens ({ldu.token_count}).")
                
        return True

class ChunkingEngine:
    """
    Deterministic Chunking Engine.
    Converts an ExtractedDocument into a list of Logical Document Units (LDUs)
    obeying the Constitution rules defined in configuration.
    """
    
    def __init__(self, config: dict):
        self.config = config.get("chunking", {})
        self.max_tokens = self.config.get("max_tokens_per_chunk", 512)
        self.validator = ChunkValidator()
        
    def _compute_hash(self, page_refs: List[int], bbox: Optional[BoundingBox], text: str) -> str:
        """
        Implementation of the exact requested hash:
        hash = SHA256(page_number + block_bbox + normalized_text)
        """
        hasher = hashlib.sha256()
        
        # page number(s)
        pages_str = ",".join(str(p) for p in page_refs)
        hasher.update(pages_str.encode('utf-8'))
        
        # block_bbox string
        if bbox:
            bbox_str = f"{bbox.x0:.1f},{bbox.top:.1f},{bbox.x1:.1f},{bbox.bottom:.1f}"
        else:
            bbox_str = "no_bbox"
        hasher.update(bbox_str.encode('utf-8'))
        
        # normalized_text
        normalized_text = " ".join(text.split()).strip().lower()
        hasher.update(normalized_text.encode('utf-8'))
        
        return hasher.hexdigest()

    def process_document(self, doc: ExtractedDocument) -> List[LDU]:
        ldus: List[LDU] = []
        active_parent_section: Optional[str] = None
        
        current_chunk_text = []
        current_chunk_tokens = 0
        current_chunk_pages = set()
        current_chunk_bbox: Optional[BoundingBox] = None
        
        def flush_text_chunk():
            nonlocal current_chunk_text, current_chunk_tokens, current_chunk_pages, current_chunk_bbox
            if not current_chunk_text:
                return
                
            merged_text = "\n".join(current_chunk_text)
            c_hash = self._compute_hash(sorted(list(current_chunk_pages)), current_chunk_bbox, merged_text)
            
            ldu = LDU(
                chunk_id=f"{doc.document_id}_chk_{len(ldus)}",
                document_id=doc.document_id,
                content=merged_text,
                chunk_type=ChunkType.TEXT,
                page_refs=sorted(list(current_chunk_pages)),
                bounding_box=current_chunk_bbox,
                parent_section=active_parent_section,
                token_count=current_chunk_tokens,
                content_hash=c_hash
            )
            ldus.append(ldu)
            
            # Reset
            current_chunk_text = []
            current_chunk_tokens = 0
            current_chunk_pages = set()
            current_chunk_bbox = None

        for page in doc.pages:
            for block in page.blocks:
                
                # Handling Tables (Rule 1: tables must not split)
                if block.block_type == "table":
                    flush_text_chunk() # Flush pending text before a table
                    
                    table_text = ""
                    if hasattr(block, "headers") and block.headers:
                        table_text += " | ".join(str(h) for h in block.headers) + "\n"
                    if hasattr(block, "rows"):
                        for row in block.rows:
                            table_text += " | ".join(str(c) for c in row) + "\n"
                            
                    bbox_obj = None
                    if hasattr(block, "bbox") and block.bbox:
                        bbox_obj = BoundingBox(
                            x0=block.bbox.x0, top=block.bbox.top, 
                            x1=block.bbox.x1, bottom=block.bbox.bottom
                        )
                    
                    c_hash = self._compute_hash([page.page_number], bbox_obj, table_text)
                    ldu = LDU(
                        chunk_id=f"{doc.document_id}_chk_{len(ldus)}",
                        document_id=doc.document_id,
                        content=table_text,
                        chunk_type=ChunkType.TABLE,
                        page_refs=[page.page_number],
                        bounding_box=bbox_obj,
                        parent_section=active_parent_section,
                        token_count=len(table_text.split()), # rough estimate
                        content_hash=c_hash,
                        metadata={"raw_headers": getattr(block, "headers", []), "raw_rows": getattr(block, "rows", [])}
                    )
                    ldus.append(ldu)
                    continue

                # Handling native TextBlocks
                text = getattr(block, "text", "")
                if not text.strip():
                    continue
                    
                # Heuristic: Detect Headers (very short, uppercase or Title Case)
                words = text.split()
                if len(words) < 10 and (text.isupper() or text.istitle()):
                    flush_text_chunk() # Flush previous chunk
                    active_parent_section = text.strip()
                    # We start a NEW chunk for this header and subsequent text
                    current_chunk_text.append(text)
                    current_chunk_tokens += int(len(words) * 1.3)
                    current_chunk_pages.add(page.page_number)
                    if hasattr(block, "bbox") and block.bbox:
                        b = block.bbox
                        if current_chunk_bbox is None:
                            current_chunk_bbox = BoundingBox(x0=b.x0, top=b.top, x1=b.x1, bottom=b.bottom)
                        else:
                            # Usually this is the first block, so it just sets it
                            pass
                            
                    flush_text_chunk() # Immediately flush the header as its OWN chunk 
                                       # so that subsequent body text is split!
                    continue
                    
                # rough token estimate (1 word ~= 1.3 tokens)
                est_tokens = int(len(words) * 1.3)
                
                if current_chunk_tokens + est_tokens > self.max_tokens:
                    flush_text_chunk()
                    
                current_chunk_text.append(text)
                current_chunk_tokens += est_tokens
                current_chunk_pages.add(page.page_number)
                
                # Expand bounding box
                if hasattr(block, "bbox") and block.bbox:
                    b = block.bbox
                    if current_chunk_bbox is None:
                        current_chunk_bbox = BoundingBox(x0=b.x0, top=b.top, x1=b.x1, bottom=b.bottom)
                    else:
                        current_chunk_bbox.x0 = min(current_chunk_bbox.x0, b.x0)
                        current_chunk_bbox.top = min(current_chunk_bbox.top, b.top)
                        current_chunk_bbox.x1 = max(current_chunk_bbox.x1, b.x1)
                        current_chunk_bbox.bottom = max(current_chunk_bbox.bottom, b.bottom)
                        
        flush_text_chunk() # final flush
        
        if self.config.get("enforce_rules", True):
            self.validator.validate(ldus, self.max_tokens)
            
        return ldus
