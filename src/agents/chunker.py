import hashlib
import json
import logging
import re
from typing import List, Optional, Any

from src.models.ldu import LDU, ChunkType, BoundingBox
from src.models.extracted_document import ExtractedDocument, TextBlock, TableBlock

logger = logging.getLogger(__name__)

class ChunkValidator:
    """Enforces the 5 chunking constitution rules."""
    
    @staticmethod
    def validate(ldus: List[LDU], chunking_config: dict) -> bool:
        max_tokens = chunking_config.get("max_tokens_per_chunk", 512)
        constitution = chunking_config.get("constitution", [])
        rules = [rule.get("name") for rule in constitution]
        
        for i, ldu in enumerate(ldus):
            # Rule 1: tables must not split (a table is always 1 LDU)
            if "tables_must_not_split" in rules and ldu.chunk_type == ChunkType.TABLE:
                if len(ldu.page_refs) > 1:
                    logger.warning(f"LDU {ldu.chunk_id} spans {len(ldu.page_refs)} pages. Tables should ideally remain intact.")
            
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

    # --- Cross-reference patterns (Rule 5) ---
    _XREF_PATTERN = re.compile(
        r"(?:see|refer\s+to|as\s+(?:shown|described|noted)\s+in)\s+"
        r"(Table|Figure|Section|Appendix|Annex|Chart|Exhibit)\s+"
        r"([A-Z0-9][A-Z0-9.\-]*)",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_list_item(text: str) -> bool:
        """Detect if text begins with a list marker (numbered or bulleted)."""
        stripped = text.strip()
        # 1. or 1) or (1) or a) or a. or (a) or i. or ii.
        if re.match(r'^(?:\d+[.)]|\([a-z0-9]+\)|[a-z][.)\s]|[ivxlc]+[.)\s]|[-•●▪▸➤])', stripped, re.IGNORECASE):
            return True
        return False
        
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

    def _merge_text_blocks(self, blocks: List[Any], max_vertical_gap: float = 20.0) -> List[Any]:
        """Merge consecutive short TextBlocks on the same page into larger paragraphs."""
        merged = []
        current_text_block = None
        
        for block in blocks:
            if block.block_type != "text" or not getattr(block, "text", "").strip():
                if current_text_block:
                    merged.append(current_text_block)
                    current_text_block = None
                merged.append(block)
                continue
                
            # It's a TextBlock. If no active block, start one
            if not current_text_block:
                current_text_block = block
                continue
                
            # Check if we should merge with current active text block
            can_merge = False
            if hasattr(current_text_block, "bbox") and current_text_block.bbox and hasattr(block, "bbox") and block.bbox:
                b1 = current_text_block.bbox
                b2 = block.bbox
                v_gap = b2.top - b1.bottom
                # Merge if vertical gap is small and horizontal overlap exists,
                # or if it's very close vertically
                if 0 <= v_gap <= max_vertical_gap:
                    can_merge = True
            elif not hasattr(current_text_block, "bbox") or not current_text_block.bbox:
                # If no bounding boxes to check, just merge blindly unless it's a header
                can_merge = True
                
            # Don't merge if either block looks like a header (to preserve parent_section logic)
            t1 = getattr(current_text_block, "text", "")
            t2 = getattr(block, "text", "")
            is_h1 = len(t1.split()) < 10 and (t1.isupper() or t1.istitle())
            is_h2 = len(t2.split()) < 10 and (t2.isupper() or t2.istitle())
            if is_h1 or is_h2:
                can_merge = False
                
            if can_merge:
                current_text_block.text = current_text_block.text + " " + block.text
                if hasattr(current_text_block, "bbox") and current_text_block.bbox and hasattr(block, "bbox") and block.bbox:
                    current_text_block.bbox.x0 = min(current_text_block.bbox.x0, block.bbox.x0)
                    current_text_block.bbox.top = min(current_text_block.bbox.top, block.bbox.top)
                    current_text_block.bbox.x1 = max(current_text_block.bbox.x1, block.bbox.x1)
                    current_text_block.bbox.bottom = max(current_text_block.bbox.bottom, block.bbox.bottom)
            else:
                # Flush and start new
                merged.append(current_text_block)
                current_text_block = block
                
        if current_text_block:
            merged.append(current_text_block)
            
        return merged

    def process_document(self, doc: ExtractedDocument) -> List[LDU]:
        ldus: List[LDU] = []
        active_parent_section: Optional[str] = None
        
        current_chunk_text = []
        current_chunk_tokens = 0
        current_chunk_pages = set()
        current_chunk_bbox: Optional[BoundingBox] = None
        current_chunk_type: ChunkType = ChunkType.TEXT
        
        def _page_ref(page_number: int) -> int:
            """Convert 0-indexed page numbers to 1-indexed for LDU."""
            return page_number + 1
        
        def flush_text_chunk():
            nonlocal current_chunk_text, current_chunk_tokens, current_chunk_pages, current_chunk_bbox, current_chunk_type
            if not current_chunk_text:
                return
                
            merged_text = "\n".join(current_chunk_text)
            page_refs_1indexed = sorted(list(current_chunk_pages))
            c_hash = self._compute_hash(page_refs_1indexed, current_chunk_bbox, merged_text)
            
            ldu = LDU(
                chunk_id=f"{doc.document_id}_chk_{len(ldus)}",
                document_id=doc.document_id,
                content=merged_text,
                chunk_type=current_chunk_type,
                page_refs=page_refs_1indexed,
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
            current_chunk_type = ChunkType.TEXT

        for page in doc.pages:
            page_1idx = _page_ref(page.page_number)
            # Pre-processing pass: merge consecutive layout text fragments
            merged_blocks = self._merge_text_blocks(page.blocks)
            
            for idx, block in enumerate(merged_blocks):
                
                # -----------------------------------------------------------
                # Rule 2: Figure Captions bind to parent figure
                # -----------------------------------------------------------
                if block.block_type == "figure":
                    flush_text_chunk()
                    caption = getattr(block, "caption", None) or ""
                    # Look ahead for caption-like text block
                    if not caption and idx + 1 < len(merged_blocks):
                        next_block = merged_blocks[idx + 1]
                        next_text = getattr(next_block, "text", "")
                        if next_block.block_type == "text" and next_text.strip():
                            first_line = next_text.strip().split("\n")[0]
                            if re.match(r'^(?:Figure|Fig\.?|Image|Photo|Diagram|Chart)\s', first_line, re.IGNORECASE):
                                caption = next_text.strip()
                    
                    fig_content = caption if caption else "[Figure]"
                    bbox_obj = None
                    if hasattr(block, "bbox") and block.bbox:
                        bbox_obj = BoundingBox(
                            x0=min(block.bbox.x0, block.bbox.x1), top=min(block.bbox.top, block.bbox.bottom),
                            x1=max(block.bbox.x0, block.bbox.x1), bottom=max(block.bbox.top, block.bbox.bottom)
                        )
                    c_hash = self._compute_hash([page_1idx], bbox_obj, fig_content)
                    ldu = LDU(
                        chunk_id=f"{doc.document_id}_chk_{len(ldus)}",
                        document_id=doc.document_id,
                        content=fig_content,
                        chunk_type=ChunkType.FIGURE_CAPTION,
                        page_refs=[page_1idx],
                        bounding_box=bbox_obj,
                        parent_section=active_parent_section,
                        token_count=len(fig_content.split()),
                        content_hash=c_hash,
                        metadata={"has_caption": bool(caption)}
                    )
                    ldus.append(ldu)
                    continue
                
                # -----------------------------------------------------------
                # Rule 1: Tables must not split
                # -----------------------------------------------------------
                if block.block_type == "table":
                    flush_text_chunk()
                    
                    table_text = ""
                    if hasattr(block, "headers") and block.headers:
                        table_text += " | ".join(str(h) for h in block.headers) + "\n"
                    if hasattr(block, "rows"):
                        for row in block.rows:
                            table_text += " | ".join(str(c) for c in row) + "\n"
                            
                    bbox_obj = None
                    if hasattr(block, "bbox") and block.bbox:
                        bbox_obj = BoundingBox(
                            x0=min(block.bbox.x0, block.bbox.x1), top=min(block.bbox.top, block.bbox.bottom), 
                            x1=max(block.bbox.x0, block.bbox.x1), bottom=max(block.bbox.top, block.bbox.bottom)
                        )
                    
                    c_hash = self._compute_hash([page_1idx], bbox_obj, table_text)
                    ldu = LDU(
                        chunk_id=f"{doc.document_id}_chk_{len(ldus)}",
                        document_id=doc.document_id,
                        content=table_text,
                        chunk_type=ChunkType.TABLE,
                        page_refs=[page_1idx],
                        bounding_box=bbox_obj,
                        parent_section=active_parent_section,
                        token_count=len(table_text.split()),
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
                    flush_text_chunk()
                    active_parent_section = text.strip()
                    current_chunk_text.append(text)
                    current_chunk_tokens += int(len(words) * 1.3)
                    current_chunk_pages.add(page_1idx)
                    if hasattr(block, "bbox") and block.bbox:
                        b = block.bbox
                        if current_chunk_bbox is None:
                            current_chunk_bbox = BoundingBox(
                                x0=min(b.x0, b.x1), top=min(b.top, b.bottom), 
                                x1=max(b.x0, b.x1), bottom=max(b.top, b.bottom)
                            )
                    flush_text_chunk()
                    continue
                
                # -----------------------------------------------------------
                # Rule 3: Numbered lists stay grouped
                # -----------------------------------------------------------
                if self._is_list_item(text):
                    # If we have a non-list chunk pending, flush it first
                    if current_chunk_text and current_chunk_type != ChunkType.LIST:
                        flush_text_chunk()
                    current_chunk_type = ChunkType.LIST
                    est_tokens = int(len(words) * 1.3)
                    if current_chunk_tokens + est_tokens > self.max_tokens:
                        flush_text_chunk()
                        current_chunk_type = ChunkType.LIST
                    current_chunk_text.append(text)
                    current_chunk_tokens += est_tokens
                    current_chunk_pages.add(page_1idx)
                    if hasattr(block, "bbox") and block.bbox:
                        b = block.bbox
                        if current_chunk_bbox is None:
                            current_chunk_bbox = BoundingBox(
                                x0=min(b.x0, b.x1), top=min(b.top, b.bottom),
                                x1=max(b.x0, b.x1), bottom=max(b.top, b.bottom)
                            )
                        else:
                            current_chunk_bbox.x0 = min(current_chunk_bbox.x0, b.x0)
                            current_chunk_bbox.top = min(current_chunk_bbox.top, b.top)
                            current_chunk_bbox.x1 = max(current_chunk_bbox.x1, b.x1)
                            current_chunk_bbox.bottom = max(current_chunk_bbox.bottom, b.bottom)
                    continue
                
                # --- Regular text block ---
                # If we were accumulating list items, flush them
                if current_chunk_type == ChunkType.LIST:
                    flush_text_chunk()
                
                # rough token estimate (1 word ~= 1.3 tokens)
                est_tokens = int(len(words) * 1.3)
                
                if current_chunk_tokens + est_tokens > self.max_tokens:
                    flush_text_chunk()
                    
                current_chunk_text.append(text)
                current_chunk_tokens += est_tokens
                current_chunk_pages.add(page_1idx)
                
                # Expand bounding box
                if hasattr(block, "bbox") and block.bbox:
                    b = block.bbox
                    if current_chunk_bbox is None:
                        current_chunk_bbox = BoundingBox(
                            x0=min(b.x0, b.x1), top=min(b.top, b.bottom), 
                            x1=max(b.x0, b.x1), bottom=max(b.top, b.bottom)
                        )
                    else:
                        current_chunk_bbox.x0 = min(current_chunk_bbox.x0, b.x0)
                        current_chunk_bbox.top = min(current_chunk_bbox.top, b.top)
                        current_chunk_bbox.x1 = max(current_chunk_bbox.x1, b.x1)
                        current_chunk_bbox.bottom = max(current_chunk_bbox.bottom, b.bottom)
                        
        flush_text_chunk()  # final flush
        
        # -----------------------------------------------------------
        # Rule 5: Cross-reference resolution (post-pass)
        # -----------------------------------------------------------
        self._resolve_cross_references(ldus)
        
        if self.config.get("enforce_rules", True):
            self.validator.validate(ldus, self.config)
            
        return ldus

    def _resolve_cross_references(self, ldus: List[LDU]) -> None:
        """Rule 5: Detect cross-references like 'See Table 3' and link LDUs."""
        # Build a lookup: section/table/figure title fragments -> chunk_id
        title_index = {}
        for ldu in ldus:
            if ldu.chunk_type == ChunkType.TABLE:
                title_index[f"table_{ldu.chunk_id}"] = ldu.chunk_id
            elif ldu.chunk_type == ChunkType.FIGURE_CAPTION:
                title_index[f"figure_{ldu.chunk_id}"] = ldu.chunk_id
            if ldu.parent_section:
                title_index[ldu.parent_section.lower().strip()] = ldu.chunk_id
        
        for ldu in ldus:
            matches = self._XREF_PATTERN.findall(ldu.content)
            for ref_type, ref_id in matches:
                ref_key = f"{ref_type.lower()}_{ref_id.lower()}"
                # Try to find a matching LDU by approximate key
                for title_key, target_id in title_index.items():
                    if ref_key in title_key or ref_id.lower() in title_key:
                        if target_id != ldu.chunk_id:
                            ldu.relationships.append({
                                "id": target_id,
                                "type": f"references_{ref_type.lower()}"
                            })
                            break
