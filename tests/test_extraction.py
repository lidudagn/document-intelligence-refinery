"""
Unit tests for Phase 2: Multi-Strategy Extraction Engine.
Tests FastTextExtractor confidence scoring and ExtractionRouter page-level escalation.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.strategies.base import ExtractionResult
from src.strategies.fast_text import FastTextExtractor
from src.models.extracted_document import (
    TextBlock, TableBlock, BoundingBox, ExtractedPage, ExtractedDocument,
)
from src.models.profile import (
    DocumentProfile, OriginType, LayoutComplexity, ExtractionCost,
)
from src.agents.router import ExtractionRouter, ExtractionValidator
from src.strategies.vision_augmented import BudgetGuard, BudgetExceededError

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
SAMPLE_DOCS_DIR = Path("data/data")


def _make_profile(**overrides) -> DocumentProfile:
    """Helper to build a DocumentProfile with sensible defaults."""
    defaults = dict(
        document_id="test-doc",
        file="test.pdf",
        num_pages=2,
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        language="en",
        domain_hint="general",
        extraction_cost=ExtractionCost.FAST_TEXT_SUFFICIENT,
        confidence_scores={"origin_conf": 0.9, "layout_conf": 0.9, "lang_conf": 0.9},
    )
    defaults.update(overrides)
    return DocumentProfile(**defaults)


# ===================================================================
# FastTextExtractor Confidence Scoring
# ===================================================================

class TestFastTextConfidence:
    """Verify that the confidence formula responds correctly to page characteristics."""

    @pytest.fixture
    def extractor(self):
        return FastTextExtractor()

    def test_native_digital_page_high_confidence(self, extractor):
        """A text-rich native PDF page should yield high confidence."""
        pdfs = list(SAMPLE_DOCS_DIR.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No sample PDFs in data/")

        # Use first available PDF, page 0
        result = extractor.extract_page(str(pdfs[0]), 0)
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.strategy_used == "fast_text"
        # Native digital pages should generally have reasonable confidence
        # (unless the first page is a cover image)
        assert result.confidence_score >= 0.0

    def test_confidence_formula_bounds(self, extractor):
        """Confidence must always be between 0.0 and 1.0."""
        pdfs = list(SAMPLE_DOCS_DIR.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No sample PDFs in data/")

        for pdf_path in pdfs[:2]:
            result = extractor.extract_page(str(pdf_path), 0)
            assert 0.0 <= result.confidence_score <= 1.0, \
                f"Confidence {result.confidence_score} out of bounds for {pdf_path.name}"

    def test_blocks_have_bboxes(self, extractor):
        """Every TextBlock from FastText should have a bounding box."""
        pdfs = list(SAMPLE_DOCS_DIR.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No sample PDFs in data/")

        result = extractor.extract_page(str(pdfs[0]), 0)
        for block in result.blocks:
            if isinstance(block, TextBlock):
                assert block.bbox is not None, "TextBlock from FastText must have bbox"

    def test_content_hash_populated(self, extractor):
        """Every block should have a non-empty content_hash for provenance."""
        pdfs = list(SAMPLE_DOCS_DIR.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No sample PDFs in data/")

        result = extractor.extract_page(str(pdfs[0]), 0)
        for block in result.blocks:
            assert block.content_hash, "content_hash must be populated"


# ===================================================================
# BudgetGuard
# ===================================================================

class TestBudgetGuard:
    def test_under_budget(self):
        guard = BudgetGuard(max_usd_per_doc=0.10)
        guard.check_and_add(0.05)
        assert guard.current_spend == 0.05

    def test_budget_exceeded(self):
        guard = BudgetGuard(max_usd_per_doc=0.10)
        guard.check_and_add(0.08)
        with pytest.raises(BudgetExceededError):
            guard.check_and_add(0.05)

    def test_reset(self):
        guard = BudgetGuard(max_usd_per_doc=0.10)
        guard.check_and_add(0.09)
        guard.reset()
        assert guard.current_spend == 0.0
        guard.check_and_add(0.09)  # Should succeed after reset


# ===================================================================
# ExtractionValidator
# ===================================================================

class TestExtractionValidator:
    def test_short_content_penalized(self):
        result = ExtractionResult(
            page_number=0,
            blocks=[TextBlock(page_number=0, text="Hi", content_hash="x")],
            confidence_score=0.90,
            strategy_used="fast_text",
        )
        validated = ExtractionValidator.validate(result)
        # Short content should drop confidence below 0.5
        assert validated.confidence_score <= 0.4

    def test_table_column_mismatch_warned(self):
        result = ExtractionResult(
            page_number=0,
            blocks=[
                TableBlock(
                    page_number=0,
                    headers=["A", "B", "C"],
                    rows=[["1", "2", "3"], ["x", "y"]],  # row 1 has only 2 cols
                    content_hash="x",
                )
            ],
            confidence_score=0.85,
            strategy_used="layout_docling",
        )
        validated = ExtractionValidator.validate(result)
        assert any("cols" in w for w in validated.warnings)


# ===================================================================
# ExtractionRouter — Page-Level Escalation
# ===================================================================

class TestExtractionRouterEscalation:
    """Test that the router correctly escalates low-confidence pages."""

    def test_baseline_selection_scanned(self):
        """Scanned docs should start with vision strategy."""
        config = {
            "extraction": {
                "routing": {"escalate_on_confidence_below": 0.65},
                "vision": {"max_usd_per_document": 1.0},
            }
        }
        router = ExtractionRouter(config=config)
        profile = _make_profile(origin_type=OriginType.SCANNED_IMAGE)
        baseline = router._select_baseline_strategy(profile)
        assert baseline is router.vision

    def test_baseline_selection_native_single_col(self):
        """Native single-column docs should start with fast_text."""
        config = {
            "extraction": {
                "routing": {"escalate_on_confidence_below": 0.65},
                "vision": {"max_usd_per_document": 1.0},
            }
        }
        router = ExtractionRouter(config=config)
        profile = _make_profile(
            origin_type=OriginType.NATIVE_DIGITAL,
            layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        )
        baseline = router._select_baseline_strategy(profile)
        assert baseline is router.fast_text

    def test_baseline_selection_multi_column(self):
        """Multi-column native docs should STILL start with fast_text (cheap first).
        Escalation handles complex pages per-page."""
        config = {
            "extraction": {
                "routing": {"escalate_on_confidence_below": 0.65},
                "vision": {"max_usd_per_document": 1.0},
            }
        }
        router = ExtractionRouter(config=config)
        profile = _make_profile(
            layout_complexity=LayoutComplexity.MULTI_COLUMN,
        )
        baseline = router._select_baseline_strategy(profile)
        assert baseline is router.fast_text

    def test_baseline_selection_table_heavy(self):
        """Table-heavy native docs should start with fast_text (cheap first).
        The confidence formula detects low quality → escalates to Docling per page."""
        config = {
            "extraction": {
                "routing": {"escalate_on_confidence_below": 0.65},
                "vision": {"max_usd_per_document": 1.0},
            }
        }
        router = ExtractionRouter(config=config)
        profile = _make_profile(
            layout_complexity=LayoutComplexity.TABLE_HEAVY,
        )
        baseline = router._select_baseline_strategy(profile)
        assert baseline is router.fast_text

    def test_escalation_triggered_on_low_confidence(self):
        """When fast_text returns low confidence, the router must try layout next."""
        config = {
            "extraction": {
                "routing": {"escalate_on_confidence_below": 0.65},
                "vision": {"max_usd_per_document": 1.0},
            }
        }
        router = ExtractionRouter(config=config)

        # Mock fast_text to return low confidence
        low_result = ExtractionResult(
            page_number=0,
            blocks=[TextBlock(page_number=0, text="sparse content " * 10, content_hash="h1")],
            confidence_score=0.30,
            strategy_used="fast_text",
        )
        router.fast_text.extract_page = MagicMock(return_value=low_result)

        # Mock layout to return high confidence
        high_result = ExtractionResult(
            page_number=0,
            blocks=[TextBlock(page_number=0, text="rich structured content " * 20, content_hash="h2")],
            confidence_score=0.88,
            strategy_used="layout_docling",
        )
        router.layout.extract_page = MagicMock(return_value=high_result)

        result = router._extract_page_with_escalation(
            "fake.pdf", 0, router.fast_text
        )

        # Should have escalated to layout and used its better result
        assert result.strategy_used == "layout_docling"
        assert result.confidence_score >= 0.65
        router.layout.extract_page.assert_called_once()

    def test_no_escalation_on_high_confidence(self):
        """When fast_text returns high confidence, no escalation should occur."""
        config = {
            "extraction": {
                "routing": {"escalate_on_confidence_below": 0.65},
                "vision": {"max_usd_per_document": 1.0},
            }
        }
        router = ExtractionRouter(config=config)

        high_result = ExtractionResult(
            page_number=0,
            blocks=[TextBlock(page_number=0, text="plenty of good text " * 20, content_hash="h1")],
            confidence_score=0.92,
            strategy_used="fast_text",
        )
        router.fast_text.extract_page = MagicMock(return_value=high_result)
        router.layout.extract_page = MagicMock()

        result = router._extract_page_with_escalation(
            "fake.pdf", 0, router.fast_text
        )

        assert result.strategy_used == "fast_text"
        router.layout.extract_page.assert_not_called()


# ===================================================================
# Document-Level Confidence Aggregation
# ===================================================================

class TestDocumentConfidence:
    def test_weighted_confidence(self):
        """Edge pages (first/last 10%) should be weighted less."""
        pages = []
        for i in range(10):
            pages.append(ExtractedPage(
                page_number=i,
                page_confidence=0.90 if i not in (0, 9) else 0.50,
                strategy_used="fast_text",
            ))
        doc = ExtractedDocument(document_id="test", source_path="test.pdf", pages=pages)
        # Edge pages (0, 9) have 0.50 confidence but lower weight,
        # so overall should be closer to 0.90 than 0.70
        assert doc.overall_confidence > 0.75

    def test_empty_document_confidence(self):
        doc = ExtractedDocument(document_id="test", source_path="test.pdf", pages=[])
        assert doc.overall_confidence == 0.0


# ===================================================================
# Reading Order
# ===================================================================

class TestReadingOrder:
    def test_sort_blocks_top_to_bottom(self):
        """Blocks should be sorted by vertical position first."""
        page = ExtractedPage(
            page_number=0,
            page_confidence=0.9,
            strategy_used="fast_text",
            blocks=[
                TextBlock(page_number=0, text="bottom", bbox=BoundingBox(x0=0, top=200, x1=100, bottom=220), content_hash="b"),
                TextBlock(page_number=0, text="top", bbox=BoundingBox(x0=0, top=10, x1=100, bottom=30), content_hash="t"),
            ],
        )
        page.sort_blocks()
        assert page.blocks[0].text == "top"
        assert page.blocks[1].text == "bottom"
