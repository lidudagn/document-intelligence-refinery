import time
import json
from pathlib import Path
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter

from src.strategies.base import BaseExtractor, ExtractionResult
from src.models.extracted_document import (
    BlockElement, TextBlock, TableBlock, FigureBlock, BoundingBox, ExtractedDocument
)

class DoclingDocumentAdapter:
    """Forms the bridge between Docling native JSON and our ExtractedDocument schema."""
    
    @staticmethod
    def parse(docling_doc: Any, page_num: int) -> List[BlockElement]:
        """
        Docling produces a unified document. We extract blocks specific to `page_num`.
        Note: docling page numbers are 1-indexed, pdfplumber is 0-indexed.
        """
        blocks: List[BlockElement] = []
        doc_page_num = page_num + 1
        
        # Iterate over texts
        for text_item in docling_doc.texts:
            # Docling might group items, we check prov for page mapping
            for prov in text_item.prov:
                if prov.page_no == doc_page_num:
                    bbox = None
                    if prov.bbox:
                        bbox = BoundingBox(
                            x0=prov.bbox.l,
                            top=prov.bbox.t,
                            x1=prov.bbox.r,
                            bottom=prov.bbox.b
                        )
                    
                    block = TextBlock(
                        page_number=page_num,
                        bbox=bbox,
                        text=text_item.text,
                        content_hash=""
                    )
                    block.content_hash = block.generate_hash(text_item.text)
                    blocks.append(block)
                    break # Usually one primary prov per item per page
                    
        # Iterate over tables
        for table_item in docling_doc.tables:
            for prov in table_item.prov:
                if prov.page_no == doc_page_num:
                    bbox = None
                    if prov.bbox:
                        bbox = BoundingBox(
                            x0=prov.bbox.l,
                            top=prov.bbox.t,
                            x1=prov.bbox.r,
                            bottom=prov.bbox.b
                        )
                    
                    # Convert Docling table to headers and rows
                    headers = []
                    rows = []
                    try:
                        table_data = table_item.export_to_dict()
                        # Simple mapping, real docling tables have complex cell matrices
                        # which are handled heavily. For phase 2, we just extract cell text.
                        # Docling export_to_dict varies by version, let's assume it has 'data'
                        grid = table_item.data.grid if hasattr(table_item, 'data') else []
                        if grid:
                            for i, row in enumerate(grid):
                                row_texts = [cell.text for cell in row]
                                if i == 0:
                                    headers = row_texts
                                else:
                                    rows.append(row_texts)
                    except Exception:
                        pass # Fallback to standard html if possible
                        
                    html_repr = table_item.export_to_html() if hasattr(table_item, 'export_to_html') else None
                    
                    block = TableBlock(
                        page_number=page_num,
                        bbox=bbox,
                        headers=headers,
                        rows=rows,
                        html_representation=html_repr,
                        content_hash=""
                    )
                    block.content_hash = block.generate_hash(str(headers) + str(rows))
                    blocks.append(block)
                    break
                    
        # Iterate over figures
        for fig_item in docling_doc.pictures:
            for prov in fig_item.prov:
                if prov.page_no == doc_page_num:
                    bbox = None
                    if prov.bbox:
                        bbox = BoundingBox(
                            x0=prov.bbox.l,
                            top=prov.bbox.t,
                            x1=prov.bbox.r,
                            bottom=prov.bbox.b
                        )
                        
                    caption = None
                    if hasattr(fig_item, 'caption') and fig_item.caption:
                        caption = fig_item.caption.text
                        
                    block = FigureBlock(
                        page_number=page_num,
                        bbox=bbox,
                        caption=caption,
                        content_hash=""
                    )
                    block.content_hash = block.generate_hash(caption or "figure")
                    blocks.append(block)
                    break
                    
        return blocks

class LayoutExtractor(BaseExtractor):
    """
    Strategy B: Medium Cost
    Uses Docling to extract high-fidelity structure, particularly tables and columns.
    Due to Docling's nature, it usually processes the whole document at once. 
    To satisfy the `extract_page` interface without redundant processing, we cache the Docling conversion.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.converter = DocumentConverter()
        self._docling_cache = {}
        
    def _get_docling_doc(self, pdf_path: str):
        if pdf_path not in self._docling_cache:
            print(f"Docling converting entirely: {pdf_path}")
            result = self.converter.convert(pdf_path)
            self._docling_cache[pdf_path] = result.document
        return self._docling_cache[pdf_path]
        
    def extract_page(self, pdf_path: str, page_num: int) -> ExtractionResult:
        start_time = time.time()
        
        try:
            docling_doc = self._get_docling_doc(pdf_path)
            blocks = DoclingDocumentAdapter.parse(docling_doc, page_num)
            
            # Since Docling uses vision+layout models, confidence is generally high
            # We assume a base confidence of 0.90 but would deduct if text blocks are unexpectedly empty
            confidence = 0.90
            
            if not blocks:
                confidence = 0.0 # Failed to extract anything from the layout model
                
            processing_time = time.time() - start_time
            return ExtractionResult(
                page_number=page_num,
                blocks=blocks,
                confidence_score=confidence,
                strategy_used="layout_docling",
                cost_estimate=0.01, # Roughly 1 cent per page compute eq.
                processing_time=processing_time
            )
            
        except Exception as e:
            return ExtractionResult(
                page_number=page_num,
                blocks=[],
                confidence_score=0.0,
                strategy_used="layout_docling",
                cost_estimate=0.0,
                processing_time=time.time() - start_time,
                warnings=[f"Docling Adapter Error: {str(e)}"]
            )
