from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field
import hashlib

class BoundingBox(BaseModel):
    """Spatial bounding box. Coordinates usually in points (1/72 inch)."""
    x0: float
    top: float
    x1: float
    bottom: float

class ExtractionBlock(BaseModel):
    """Base model for any continuous block of extracted content on a single page."""
    block_type: str
    page_number: int
    bbox: Optional[BoundingBox] = None
    content_hash: str = Field(description="SHA-256 hash of the block content for provenance tracking")

    def generate_hash(self, text_content: str):
        return hashlib.sha256(f"{self.page_number}_{text_content}".encode()).hexdigest()

class TextBlock(ExtractionBlock):
    block_type: str = "text"
    text: str

class TableBlock(ExtractionBlock):
    block_type: str = "table"
    headers: List[str]
    rows: List[List[str]]
    html_representation: Optional[str] = None # Sometimes extractors give good HTML tables

class FigureBlock(ExtractionBlock):
    block_type: str = "figure"
    caption: Optional[str] = None
    image_base64: Optional[str] = None # Or a reference path to saved image

# Type alias for all valid blocks
BlockElement = Union[TextBlock, TableBlock, FigureBlock]

class ExtractedPage(BaseModel):
    """Represents a single parsed page from a document, holding ordered blocks."""
    page_number: int
    blocks: List[BlockElement] = []
    page_confidence: float = Field(default=0.0, description="Confidence score 0.0-1.0")
    strategy_used: str = Field(..., description="E.g., fast_text, layout, vision")
    
    def sort_blocks(self):
        """Reading Order Reconstruction: sort blocks top-to-bottom, left-to-right."""
        def sorting_key(block: BlockElement):
            if not block.bbox:
                return (9999, 9999) # Send elements without bboxes to the end
            # Using top coordinate (ascending) primarily.
            # Adding a small bucket threshold for horizontal line alignment (e.g. 5 points)
            top_bucket = round(block.bbox.top / 5) * 5
            return (top_bucket, block.bbox.x0)
        
        self.blocks.sort(key=sorting_key)

class ExtractedDocument(BaseModel):
    """The unified representation returned by all extraction strategies."""
    document_id: str
    source_path: str
    pages: List[ExtractedPage] = []
    
    @property
    def total_pages(self) -> int:
        return len(self.pages)
        
    @property
    def overall_confidence(self) -> float:
        """
        Document-Level Confidence Aggregation.
        Weighted by page importance. The first 10% and last 10% of a document
        are typically covers/appendices and are weighted less heavily than core content.
        """
        if not self.pages:
            return 0.0
            
        total_weight = 0.0
        weighted_sum = 0.0
        
        n_pages = len(self.pages)
        edge_zone = max(1, int(n_pages * 0.10))
        
        for p in self.pages:
            weight = 1.0
            # De-weight edges (covers, TOCs, appendices) slightly to prevent them from overly skewing the score
            if p.page_number <= edge_zone or p.page_number > (n_pages - edge_zone):
                weight = 0.5
                
            weighted_sum += (p.page_confidence * weight)
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 0.0
