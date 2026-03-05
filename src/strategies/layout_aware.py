import time
import json
import tempfile
import logging
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF for single-page extraction

from src.strategies.base import BaseExtractor, ExtractionResult
from src.models.extracted_document import (
    BlockElement, TextBlock, TableBlock, FigureBlock, BoundingBox, ExtractedDocument
)

logger = logging.getLogger(__name__)

# Timeout for a single page Docling conversion (seconds)
DOCLING_PAGE_TIMEOUT = 120


def _docling_subprocess_worker(tmp_pdf_path: str, result_queue: multiprocessing.Queue):
    """Run Docling conversion in a subprocess to isolate memory usage.
    
    If this subprocess is OOM-killed, the parent process survives.
    Results are passed back via the multiprocessing.Queue.
    """
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(tmp_pdf_path)
        docling_doc = result.document

        # Serialize the blocks we need from the docling document
        blocks_data = []

        # Extract texts
        for text_item in docling_doc.texts:
            for prov in text_item.prov:
                if prov.page_no == 1:  # temp PDF has only 1 page
                    bbox = None
                    if prov.bbox:
                        bbox = {"x0": prov.bbox.l, "top": prov.bbox.t,
                                "x1": prov.bbox.r, "bottom": prov.bbox.b}
                    blocks_data.append({
                        "type": "text",
                        "text": text_item.text,
                        "bbox": bbox,
                    })
                    break

        # Extract tables
        for table_item in docling_doc.tables:
            for prov in table_item.prov:
                if prov.page_no == 1:
                    bbox = None
                    if prov.bbox:
                        bbox = {"x0": prov.bbox.l, "top": prov.bbox.t,
                                "x1": prov.bbox.r, "bottom": prov.bbox.b}
                    headers = []
                    rows = []
                    try:
                        grid = table_item.data.grid if hasattr(table_item, 'data') else []
                        if grid:
                            for i, row in enumerate(grid):
                                row_texts = [cell.text for cell in row]
                                if i == 0:
                                    headers = row_texts
                                else:
                                    rows.append(row_texts)
                    except Exception:
                        pass
                    html_repr = None
                    try:
                        html_repr = table_item.export_to_html() if hasattr(table_item, 'export_to_html') else None
                    except Exception:
                        pass
                    blocks_data.append({
                        "type": "table",
                        "headers": headers,
                        "rows": rows,
                        "html": html_repr,
                        "bbox": bbox,
                    })
                    break

        # Extract figures
        for fig_item in docling_doc.pictures:
            for prov in fig_item.prov:
                if prov.page_no == 1:
                    bbox = None
                    if prov.bbox:
                        bbox = {"x0": prov.bbox.l, "top": prov.bbox.t,
                                "x1": prov.bbox.r, "bottom": prov.bbox.b}
                    caption = None
                    if hasattr(fig_item, 'caption') and fig_item.caption:
                        caption = fig_item.caption.text
                    blocks_data.append({
                        "type": "figure",
                        "caption": caption,
                        "bbox": bbox,
                    })
                    break

        result_queue.put(("ok", blocks_data))
    except Exception as e:
        result_queue.put(("error", str(e)))


class LayoutExtractor(BaseExtractor):
    """
    Strategy B: Medium Cost
    Uses Docling to extract high-fidelity structure, particularly tables and columns.

    When used as an escalation target (extract_page), it extracts only the
    requested page as a temporary single-page PDF and feeds that to Docling.
    
    Docling runs in a subprocess to prevent OOM-kills from crashing the main 
    extraction pipeline.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # No longer initialize converter here — it's created inside subprocess

    def _extract_single_page_pdf(self, pdf_path: str, page_num: int) -> str:
        """Extract a single page from a PDF into a temporary file.
        Returns the path to the temporary single-page PDF."""
        src = fitz.open(pdf_path)
        tmp = fitz.open()
        tmp.insert_pdf(src, from_page=page_num, to_page=page_num)

        tmp_path = tempfile.mktemp(suffix=".pdf", prefix=f"docling_page{page_num}_")
        tmp.save(tmp_path)
        tmp.close()
        src.close()
        return tmp_path

    def _reconstruct_blocks(self, blocks_data: List[Dict], page_num: int) -> List[BlockElement]:
        """Reconstruct BlockElement objects from serialized subprocess data."""
        blocks: List[BlockElement] = []
        for bd in blocks_data:
            bbox = None
            if bd.get("bbox"):
                bbox = BoundingBox(**bd["bbox"])

            if bd["type"] == "text":
                block = TextBlock(
                    page_number=page_num,
                    bbox=bbox,
                    text=bd["text"],
                    content_hash=""
                )
                block.content_hash = block.generate_hash(bd["text"])
                blocks.append(block)
            elif bd["type"] == "table":
                block = TableBlock(
                    page_number=page_num,
                    bbox=bbox,
                    headers=bd.get("headers", []),
                    rows=bd.get("rows", []),
                    html_representation=bd.get("html"),
                    content_hash=""
                )
                block.content_hash = block.generate_hash(
                    str(bd.get("headers", [])) + str(bd.get("rows", []))
                )
                blocks.append(block)
            elif bd["type"] == "figure":
                block = FigureBlock(
                    page_number=page_num,
                    bbox=bbox,
                    caption=bd.get("caption"),
                    content_hash=""
                )
                block.content_hash = block.generate_hash(bd.get("caption") or "figure")
                blocks.append(block)
        return blocks

    def extract_page(self, pdf_path: str, page_num: int) -> ExtractionResult:
        """Extract a single page by creating a temp single-page PDF for Docling.

        Docling runs in a **subprocess** so that if a page causes an OOM-kill,
        only the subprocess dies — the main extraction pipeline survives.
        """
        start_time = time.time()
        tmp_pdf_path = None

        try:
            # Extract just this page as a temporary PDF
            tmp_pdf_path = self._extract_single_page_pdf(pdf_path, page_num)
            logger.info(f"  Docling processing single page {page_num} from {Path(pdf_path).name}")

            # Run Docling in a subprocess to isolate memory usage
            result_queue = multiprocessing.Queue()
            proc = multiprocessing.Process(
                target=_docling_subprocess_worker,
                args=(tmp_pdf_path, result_queue),
            )
            proc.start()
            proc.join(timeout=DOCLING_PAGE_TIMEOUT)

            if proc.is_alive():
                logger.warning(f"  Docling timed out on page {page_num} after {DOCLING_PAGE_TIMEOUT}s, killing subprocess")
                proc.kill()
                proc.join()
                return ExtractionResult(
                    page_number=page_num, blocks=[], confidence_score=0.0,
                    strategy_used="layout_docling", cost_estimate=0.0,
                    processing_time=time.time() - start_time,
                    warnings=[f"Docling timed out after {DOCLING_PAGE_TIMEOUT}s"]
                )

            if proc.exitcode != 0:
                logger.warning(f"  Docling subprocess crashed on page {page_num} (exit code {proc.exitcode})")
                return ExtractionResult(
                    page_number=page_num, blocks=[], confidence_score=0.0,
                    strategy_used="layout_docling", cost_estimate=0.0,
                    processing_time=time.time() - start_time,
                    warnings=[f"Docling subprocess crashed (exit={proc.exitcode})"]
                )

            # Get results from the subprocess
            if result_queue.empty():
                logger.warning(f"  Docling returned no results for page {page_num}")
                return ExtractionResult(
                    page_number=page_num, blocks=[], confidence_score=0.0,
                    strategy_used="layout_docling", cost_estimate=0.0,
                    processing_time=time.time() - start_time,
                    warnings=["Docling subprocess returned no data"]
                )

            status, data = result_queue.get_nowait()

            if status == "error":
                logger.warning(f"  Docling failed on page {page_num}: {data}")
                return ExtractionResult(
                    page_number=page_num, blocks=[], confidence_score=0.0,
                    strategy_used="layout_docling", cost_estimate=0.0,
                    processing_time=time.time() - start_time,
                    warnings=[f"Docling error: {data}"]
                )

            # Success! Reconstruct blocks from serialized data
            blocks = self._reconstruct_blocks(data, page_num)

            confidence = 0.90 if blocks else 0.0
            processing_time = time.time() - start_time
            return ExtractionResult(
                page_number=page_num,
                blocks=blocks,
                confidence_score=confidence,
                strategy_used="layout_docling",
                cost_estimate=0.01,
                processing_time=processing_time
            )

        except Exception as e:
            logger.warning(f"  Docling failed on page {page_num}: {e}")
            return ExtractionResult(
                page_number=page_num,
                blocks=[],
                confidence_score=0.0,
                strategy_used="layout_docling",
                cost_estimate=0.0,
                processing_time=time.time() - start_time,
                warnings=[f"Docling Adapter Error: {str(e)}"]
            )
        finally:
            # Clean up temp file
            if tmp_pdf_path:
                import os
                try:
                    os.unlink(tmp_pdf_path)
                except OSError:
                    pass
