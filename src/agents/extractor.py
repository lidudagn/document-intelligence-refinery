import json
import os
import time
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from src.models.profile import DocumentProfile, OriginType, LayoutComplexity, ExtractionCost
from src.models.extracted_document import ExtractedDocument, ExtractedPage, TableBlock
from src.strategies.base import BaseExtractor, ExtractionResult
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_aware import LayoutExtractor
from src.strategies.vision_augmented import VisionExtractor, BudgetGuard

import yaml
import pdfplumber

load_dotenv()
logger = logging.getLogger(__name__)


def load_extraction_config(config_path: str = "rubric/extraction_rules.yaml") -> Dict:
    """Load extraction configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ExtractionValidator:
    """
    Post-extraction structural checks to catch hallucinations and
    low-quality outputs before they propagate downstream.
    """

    @staticmethod
    def validate(result: ExtractionResult) -> ExtractionResult:
        warnings = list(result.warnings)

        # Check 1: Suspiciously short content
        total_text_len = sum(
            len(b.text) for b in result.blocks if hasattr(b, "text")
        )
        if total_text_len < 20 and result.confidence_score > 0.5:
            warnings.append(
                f"Page {result.page_number}: suspiciously short content ({total_text_len} chars) despite moderate confidence"
            )
            result.confidence_score = min(result.confidence_score, 0.4)

        # Check 2: Table column consistency
        for b in result.blocks:
            if isinstance(b, TableBlock) and b.headers and b.rows:
                expected_cols = len(b.headers)
                for i, row in enumerate(b.rows):
                    if len(row) != expected_cols:
                        warnings.append(
                            f"Page {result.page_number}: table row {i} has {len(row)} cols, expected {expected_cols}"
                        )

        result.warnings = warnings
        return result


class ExtractionLedger:
    """Appends extraction audit entries to .refinery/extraction_ledger.jsonl."""

    def __init__(self, ledger_path: str = ".refinery/extraction_ledger.jsonl"):
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: Dict[str, Any]):
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class ExtractionRouter:
    """
    Central coordinator that reads a DocumentProfile and orchestrates
    page-level extraction with confidence-gated escalation.

    Escalation chain: Strategy A (fast_text) -> Strategy B (layout) -> Strategy C (vision)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or load_extraction_config()
        extraction_cfg = self.config.get("extraction", {})
        self.escalation_threshold = extraction_cfg.get("routing", {}).get(
            "escalate_on_confidence_below", 0.65
        )

        # Create a shared BudgetGuard so cumulative spend is tracked across
        # all pages in one document (not reset per page).
        vision_cfg = extraction_cfg.get("vision", {})
        max_usd = float(os.getenv(
            "MAX_VLM_COST_PER_DOCUMENT",
            vision_cfg.get("max_usd_per_document", 0.50),
        ))
        self.budget_guard = BudgetGuard(max_usd_per_doc=max_usd)

        # FastText is always eagerly loaded — it's lightweight (just pdfplumber).
        self.fast_text = FastTextExtractor(config=self.config)

        # Layout and Vision are LAZY-LOADED. They only consume memory when a
        # page actually needs escalation. This prevents loading ~GBs of Docling
        # ML models (layout detection, table structure, OCR) for documents
        # where FastText handles most or all pages.
        self._layout: Optional[LayoutExtractor] = None
        self._vision: Optional[VisionExtractor] = None

        self.ledger = ExtractionLedger()
        self.validator = ExtractionValidator()

    @property
    def layout(self) -> LayoutExtractor:
        """Lazy-load LayoutExtractor (Docling) only when first needed."""
        if self._layout is None:
            logger.info("Lazy-loading LayoutExtractor (Docling)...")
            self._layout = LayoutExtractor(config=self.config)
        return self._layout

    @property
    def vision(self) -> VisionExtractor:
        """Lazy-load VisionExtractor (VLM API) only when first needed."""
        if self._vision is None:
            logger.info("Lazy-loading VisionExtractor (VLM)...")
            self._vision = VisionExtractor(config=self.config, budget_guard=self.budget_guard)
        return self._vision

    def _select_baseline_strategy(
        self, profile: DocumentProfile
    ) -> BaseExtractor:
        """Choose the initial strategy based on the DocumentProfile.

        Core principle: CHEAP FIRST, escalate per page only if needed.

        - native_digital (any layout): Always start with FastText. It runs in
          ~1ms/page and its confidence formula naturally detects pages that need
          escalation (table-heavy pages yield low char_density → low confidence
          → per-page escalation to Docling). This prevents sending ALL pages
          of a large document through an expensive strategy.
        - scanned_image: Skip straight to Vision — there is no text stream
          for FastText or Docling to read.
        """
        if profile.origin_type == OriginType.SCANNED_IMAGE:
            return self.vision
        # All native_digital documents start cheap, regardless of layout_complexity.
        # The escalation chain handles table_heavy/multi_column pages automatically.
        return self.fast_text

    def _get_escalation_chain(self, baseline: BaseExtractor):
        """Yield extractors to try after the baseline fails (lazy generator).

        Uses a generator so that accessing self.layout / self.vision (which are
        lazy-loaded properties) only happens when the escalation step is reached.
        This means Docling models are loaded only if FastText confidence is low,
        and Vision API client is created only if Docling also fails.
        """
        if not isinstance(baseline, LayoutExtractor):
            yield self.layout
        if not isinstance(baseline, VisionExtractor):
            yield self.vision

    def extract_document(
        self, pdf_path: str, profile: DocumentProfile
    ) -> ExtractedDocument:
        """
        Main entry point. Extracts a document page-by-page with escalation.
        """
        start_time = time.time()
        total_cost = 0.0
        page_strategy_map: Dict[int, str] = {}

        # Reset the shared budget guard for this new document
        self.budget_guard.reset()

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        baseline = self._select_baseline_strategy(profile)
        logger.info(
            f"Document {profile.document_id}: baseline strategy = {type(baseline).__name__} for {total_pages} pages"
        )

        pages: List[ExtractedPage] = []

        # Setup checkpointing directory
        chk_dir = Path(f".refinery/extractions/{profile.document_id}_pages")
        chk_dir.mkdir(parents=True, exist_ok=True)

        for page_num in range(total_pages):
            page_chk_path = chk_dir / f"page_{page_num}.json"
            
            # Try to load from checkpoint
            if page_chk_path.exists():
                try:
                    with open(page_chk_path, "r") as f:
                        page_data = json.load(f)
                    page = ExtractedPage.model_validate(page_data)
                    pages.append(page)
                    page_strategy_map[page_num] = page.strategy_used
                    logger.info(f"  Page {page_num}: loaded from checkpoint ({page.strategy_used})")
                    continue
                except Exception as e:
                    logger.warning(f"  Page {page_num}: failed to load checkpoint ({e}), re-extracting")

            try:
                result = self._extract_page_with_escalation(
                    pdf_path, page_num, baseline
                )
                total_cost += result.cost_estimate

                page = ExtractedPage(
                    page_number=result.page_number,
                    blocks=result.blocks,
                    page_confidence=result.confidence_score,
                    strategy_used=result.strategy_used,
                )
                page.sort_blocks()
                pages.append(page)
                page_strategy_map[page_num] = result.strategy_used
                
                # Save checkpoint
                with open(page_chk_path, "w") as f:
                    json.dump(page.model_dump(), f, indent=2, default=str)

            except Exception as e:
                logger.error(
                    f"  Page {page_num}: extraction crashed ({type(e).__name__}: {e}). "
                    f"Recording empty page and continuing."
                )
                page = ExtractedPage(
                    page_number=page_num,
                    blocks=[],
                    page_confidence=0.0,
                    strategy_used="failed",
                )
                pages.append(page)
                page_strategy_map[page_num] = "failed"

        doc = ExtractedDocument(
            document_id=profile.document_id,
            source_path=pdf_path,
            pages=pages,
        )

        processing_time = time.time() - start_time

        # Log to ledger
        self.ledger.log(
            {
                "document_id": profile.document_id,
                "file": profile.file,
                "total_pages": total_pages,
                "page_strategies": page_strategy_map,
                "overall_confidence": round(doc.overall_confidence, 4),
                "total_cost_usd": round(total_cost, 6),
                "processing_time_s": round(processing_time, 3),
            }
        )

        logger.info(
            f"Document {profile.document_id}: confidence={doc.overall_confidence:.3f}, "
            f"cost=${total_cost:.4f}, time={processing_time:.1f}s"
        )

        # Clean up temporary checkpoint directory on successful extraction
        if chk_dir.exists():
            shutil.rmtree(chk_dir)

        return doc

    def _extract_page_with_escalation(
        self, pdf_path: str, page_num: int, baseline: BaseExtractor
    ) -> ExtractionResult:
        """
        Extract a single page. If confidence is below threshold, escalate
        to the next strategy in the chain.
        """
        result = baseline.extract_page(pdf_path, page_num)
        result = self.validator.validate(result)

        if result.confidence_score >= self.escalation_threshold:
            return result

        # Escalate through the chain (lazy generator — each step triggers loading)
        for fallback in self._get_escalation_chain(baseline):
            logger.info(
                f"  Page {page_num}: confidence {result.confidence_score:.2f} < {self.escalation_threshold} "
                f"→ escalating to {type(fallback).__name__}"
            )
            escalated = fallback.extract_page(pdf_path, page_num)
            escalated = self.validator.validate(escalated)

            if escalated.confidence_score > result.confidence_score:
                result = escalated

            if result.confidence_score >= self.escalation_threshold:
                break

        return result
