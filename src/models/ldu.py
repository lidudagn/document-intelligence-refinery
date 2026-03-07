from pydantic import BaseModel, Field, field_validator, model_validator
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
    
    @model_validator(mode='after')
    def check_coordinates(self) -> 'BoundingBox':
        if self.x0 > self.x1:
            raise ValueError(f'x0 ({self.x0}) cannot be greater than x1 ({self.x1})')
        if self.top > self.bottom:
            raise ValueError(f'top ({self.top}) cannot be greater than bottom ({self.bottom})')
        return self

class LDU(BaseModel):
    """
    Logical Document Unit (LDU).
    A semantically coherent, self-contained chunk of text that preserves structural context.
    """
    chunk_id: str
    document_id: str
    content: str
    chunk_type: ChunkType
    page_refs: List[int] = Field(description="List of pages this LDU spans. Usually just one.")
    bounding_box: Optional[BoundingBox] = None
    parent_section: Optional[str] = Field(default=None, description="The section header that encloses this chunk.")
    token_count: int = Field(default=0, description="Estimated token size of the content.")
    content_hash: str = Field(..., description="SHA-256 spatial and content hash: SHA256(page_number + bbox_str + normalized_text)")
    
    # metadata for specific types (e.g. parent figure for captions, or actual row data for tables)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Semantic relationships (e.g., cross-references, parent-child hierarchies)
    relationships: List[dict[str, str]] = Field(
        default_factory=list, 
        description="List of related chunk IDs and their relation type (e.g. {'id': 'chunk_1', 'type': 'parent'})"
    )

    @field_validator('token_count')
    @classmethod
    def validate_token_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError('token_count cannot be negative')
        return v

    @field_validator('page_refs')
    @classmethod
    def validate_page_refs(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError('page_refs cannot be empty')
        if any(page < 1 for page in v):
            raise ValueError('page_refs must contain valid 1-indexed page numbers')
        return v

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('content cannot be empty or just whitespace')
        return v

    @field_validator('relationships')
    @classmethod
    def validate_relationships(cls, v: List[dict[str, str]]) -> List[dict[str, str]]:
        for rel in v:
            if 'id' not in rel or 'type' not in rel:
                raise ValueError('relationship dicts must have "id" and "type" keys')
        return v
