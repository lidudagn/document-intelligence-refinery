from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time

from src.models.extracted_document import BlockElement, ExtractedDocument, ExtractedPage

class ExtractionResult(BaseModel):
    """Wrapper for the output of a single page extraction."""
    page_number: int
    blocks: List[BlockElement] = []
    confidence_score: float = Field(default=0.0, description="0.0 to 1.0 confidence of extraction quality")
    strategy_used: str
    cost_estimate: float = 0.0
    processing_time: float = 0.0
    warnings: List[str] = []

class BaseExtractor(ABC):
    """Abstract interface for all extraction strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    def extract_page(self, pdf_path: str, page_num: int) -> ExtractionResult:
        """Extract a single page. page_num is 0-indexed."""
        pass
        
    def extract_document(self, pdf_path: str, document_id: str = "doc") -> ExtractedDocument:
        """Extract the entire document by iterating pages.
        Usually only the Router calls this, or tests.
        """
        import pdfplumber
        start_time = time.time()
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
        pages: List[ExtractedPage] = []
        for i in range(total_pages):
            res = self.extract_page(pdf_path, i)
            page = ExtractedPage(
                page_number=res.page_number,
                blocks=res.blocks,
                page_confidence=res.confidence_score,
                strategy_used=res.strategy_used
            )
            page.sort_blocks()
            pages.append(page)
            
        return ExtractedDocument(
            document_id=document_id,
            source_path=pdf_path,
            pages=pages
        )
