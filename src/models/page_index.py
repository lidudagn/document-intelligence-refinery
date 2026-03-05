from typing import List, Optional, Any
from pydantic import BaseModel, Field

class Section(BaseModel):
    """
    A single node in the document hierarchy tree.
    """
    title: str = Field(description="The extracted section header name")
    page_start: int
    page_end: int
    child_sections: List["Section"] = Field(default_factory=list, description="Sub-sections")
    
    # Enrichment metadata for LLM-based navigation without reading the whole section
    key_entities: List[str] = Field(default_factory=list, description="Extracted named entities (people, orgs, terms).")
    summary: str = Field(default="", description="LLM-generated 2-3 sentence summary of the section's content.")
    data_types_present: List[str] = Field(
        default_factory=list,
        description="e.g. ['tables', 'figures', 'financial_data']"
    )

# Rebuild model to allow self-referencing Section type
Section.model_rebuild()

class PageIndex(BaseModel):
    """
    The hierarchical navigation structure over the document.
    A nested tree array where each node is a Section.
    """
    document_id: str
    root_sections: List[Section] = Field(default_factory=list, description="Top level sections")
