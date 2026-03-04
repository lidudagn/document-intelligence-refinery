import json
import os
import time
import logging
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

        self.fast_text = FastTextExtractor(config=self.config)
        self.layout = LayoutExtractor(config=self.config)
        self.vision = VisionExtractor(config=self.config, budget_guard=self.budget_guard)
        self.ledger = ExtractionLedger()
        self.validator = ExtractionValidator()

    def _select_baseline_strategy(
        self, profile: DocumentProfile
    ) -> BaseExtractor:
        """Choose the initial strategy based on the DocumentProfile."""
        if profile.origin_type == OriginType.SCANNED_IMAGE:
            return self.vision
        if profile.origin_type == OriginType.NATIVE_DIGITAL and profile.layout_complexity == LayoutComplexity.SINGLE_COLUMN:
            return self.fast_text
        # For multi_column, table_heavy, mixed, figure_heavy -> layout
        return self.layout

    def _get_escalation_chain(self, baseline: BaseExtractor) -> List[BaseExtractor]:
        """Return the ordered list of extractors to try after the baseline fails."""
        chain = []
        if baseline is not self.layout:
            chain.append(self.layout)
        if baseline is not self.vision:
            chain.append(self.vision)
        return chain

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

        for page_num in range(total_pages):
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

        # Escalate through the chain
        chain = self._get_escalation_chain(baseline)
        for fallback in chain:
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
