from pydantic import BaseModel, Field
from typing import List, Optional, Any
from enum import Enum

class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    LIST = "list"
    SECTION_HEADER = "section_header"

class BoundingBox(BaseModel):
    x0: float
    top: float
    x1: float
    bottom: float

class LDU(BaseModel):
    """
    Logical Document Unit (LDU).
    A semantically coherent, self-contained chunk of text that preserves structural context.
    """
    chunk_id: str
    content: str
    chunk_type: ChunkType
    page_refs: List[int] = Field(description="List of pages this LDU spans. Usually just one.")
    bounding_box: Optional[BoundingBox] = None
    parent_section: Optional[str] = Field(default=None, description="The section header that encloses this chunk.")
    token_count: int = Field(default=0, description="Estimated token size of the content.")
    content_hash: str = Field(..., description="SHA-256 spatial and content hash for provenance.")
    
    # Metadata for specific types (e.g. parent figure for captions, or actual row data for tables)
    metadata: dict[str, Any] = Field(default_factory=dict)
