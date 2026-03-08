import time
import os
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
from dotenv import load_dotenv

from openai import OpenAI

from src.strategies.base import BaseExtractor, ExtractionResult

# Load .env file from project root
load_dotenv()
from src.models.extracted_document import BlockElement, TextBlock, TableBlock

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    pass


class BudgetGuard:
    """Enforces absolute real-world dollar limits on LLM API calls per document.
    
    Designed to be shared across pages within one document extraction.
    Call reset() before starting a new document.
    """

    def __init__(self, max_usd_per_doc: float = 0.10, warning_threshold: float = 0.80):
        self.max_usd = max_usd_per_doc
        self.warning_threshold = warning_threshold
        self.current_spend = 0.0

    @property
    def usage_ratio(self) -> float:
        """Return current spend as a fraction of the budget (0.0 to 1.0+)."""
        return self.current_spend / self.max_usd if self.max_usd > 0 else 1.0

    @property
    def is_budget_tight(self) -> bool:
        """True when spend exceeds the warning threshold (default 80%)."""
        return self.usage_ratio >= self.warning_threshold

    def check_and_add(self, usd_cost: float):
        """Add cost and raise if budget exceeded."""
        if self.current_spend + usd_cost > self.max_usd:
            raise BudgetExceededError(
                f"Extraction halted: cost ${self.current_spend + usd_cost:.4f} "
                f"exceeds max budget ${self.max_usd:.2f}"
            )
        self.current_spend += usd_cost

    def reset(self):
        """Reset spend counter for a new document."""
        self.current_spend = 0.0


# ---------------------------------------------------------------------------
# VLM Prompt Templates & Structured Output Schemas
# ---------------------------------------------------------------------------

class VLMCell(BaseModel):
    value: str = Field(description="The string content of the table cell")


class VLMRow(BaseModel):
    cells: List[VLMCell]


class VLMTable(BaseModel):
    headers: List[str] = Field(description="List of column header names")
    rows: List[VLMRow] = Field(description="List of table rows containing data cells")


class VLMTextSection(BaseModel):
    content: str = Field(description="Paragraph or continuous block of text")


class VLMPageExtraction(BaseModel):
    text_sections: List[VLMTextSection] = Field(
        default_factory=list,
        description="All paragraphs and textual narrative on the page",
    )
    tables: List[VLMTable] = Field(
        default_factory=list,
        description="Any tabular data found on the page",
    )


VISION_SYSTEM_PROMPT = """You are an expert Document Intelligence extraction system.
Your task is to extract ALL structural content from the provided page image exactly as it appears.

Rules:
1. Extract every table with strict row and column alignment. Each table must have headers and rows.
2. Extract all continuous narrative paragraphs as text sections.
3. Do NOT hallucinate content. Only extract what is visually present.
4. Ignore repeating headers/footers across pages.
5. Preserve numerical precision exactly as shown (e.g., "$4,231.50" not "$4232").
6. For scanned/degraded text, do your best OCR but mark uncertain words with [?]."""


# Pricing per 1M tokens (GPT-4o-mini via OpenRouter, as of 2024)
PRICING = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "google/gemini-flash-1.5": {"input": 0.075, "output": 0.300},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.300},
}
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.0-flash")


class VisionExtractor(BaseExtractor):
    """
    Strategy C: High Cost
    Uses OpenAI (or OpenRouter) multimodal API with structured outputs
    to parse complex, degraded, or pure-image pages.
    
    BudgetGuard is passed in externally by the ExtractionRouter to ensure
    cumulative document-level cost enforcement.
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        budget_guard: Optional[BudgetGuard] = None,
    ):
        super().__init__(config)

        # Budget: prefer env var, fall back to yaml config, default to $0.50
        max_usd = float(os.getenv(
            "MAX_VLM_COST_PER_DOCUMENT",
            self.config.get("extraction", {}).get("vision", {}).get("max_usd_per_document", 0.50),
        ))

        # Use externally-provided guard if available, otherwise create one
        self.budget_guard = budget_guard or BudgetGuard(max_usd_per_doc=max_usd)

        self.model = os.getenv("DEFAULT_LLM_MODEL", DEFAULT_MODEL)

        if "/" in self.model and os.getenv("OPENROUTER_API_KEY"):
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.getenv("OPENROUTER_API_KEY")
        elif "gemini" in self.model.lower() and os.getenv("GOOGLE_API_KEY"):
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            api_key = os.getenv("GOOGLE_API_KEY")
        else:
            base_url = None
            api_key = os.getenv("OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

    def _render_page_to_base64(self, pdf_path: str, page_num: int, zoom: float = 2.0) -> str:
        """Render a specific PDF page as a PNG image and return as base64.
        
        Args:
            zoom: Render scale factor. 2.0 = high quality, 1.0 = lower quality
                  but ~50% fewer image tokens (and thus ~50% cheaper).
        """
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(img_bytes).decode("utf-8")

    def _call_vlm_api(self, base64_image: str) -> Tuple[VLMPageExtraction, float]:
        """
        Real call to OpenAI structured-output API.
        Returns parsed extraction and USD cost.
        """
        if not self.client:
            raise RuntimeError(
                "OPENAI_API_KEY not set. VisionExtractor requires an API key "
                "to call the VLM. Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL "
                "for OpenRouter)."
            )

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        }
                    ],
                },
            ],
            response_format=VLMPageExtraction,
        )

        # Calculate real cost from usage
        usage = response.usage
        fallback_pricing = PRICING.get(DEFAULT_MODEL, {"input": 0.075, "output": 0.300})
        pricing = PRICING.get(self.model, fallback_pricing)
        cost = (
            (usage.prompt_tokens * pricing["input"] / 1_000_000)
            + (usage.completion_tokens * pricing["output"] / 1_000_000)
        )

        parsed: VLMPageExtraction = response.choices[0].message.parsed
        return parsed, cost

    def _calculate_confidence(self, blocks: List[BlockElement], warnings: List[str]) -> float:
        """
        Calibrated confidence based on what was actually extracted.
        Not hardcoded — derived from extraction results.
        """
        if not blocks:
            return 0.20  # Empty extraction penalty

        total_text = sum(len(b.text) for b in blocks if hasattr(b, "text"))
        has_tables = any(isinstance(b, TableBlock) for b in blocks)

        # Base confidence for non-empty VLM output
        confidence = 0.85

        # Bonus for substantial text
        if total_text > 200:
            confidence += 0.05

        # Bonus for structured table extraction
        if has_tables:
            confidence += 0.05

        # Penalties
        if total_text < 50:
            confidence -= 0.15  # Suspiciously little content

        # Check table integrity
        for b in blocks:
            if isinstance(b, TableBlock) and b.headers and b.rows:
                for row in b.rows:
                    if len(row) != len(b.headers):
                        confidence -= 0.10
                        warnings.append("Table column count mismatch detected in VLM output")
                        break

        return max(0.0, min(1.0, confidence))

    def extract_page(self, pdf_path: str, page_num: int) -> ExtractionResult:
        start_time = time.time()
        blocks: List[BlockElement] = []
        warnings: List[str] = []

        try:
            # Adaptive resolution: drop to 1x zoom when budget is running low
            if self.budget_guard.is_budget_tight:
                zoom = 1.0
                logger.info(
                    f"  Budget at {self.budget_guard.usage_ratio:.0%} "
                    f"(${self.budget_guard.current_spend:.4f}/${self.budget_guard.max_usd:.2f}) "
                    f"→ switching to low-res mode (zoom=1.0) to conserve budget"
                )
            else:
                zoom = 2.0

            b64_img = self._render_page_to_base64(pdf_path, page_num, zoom=zoom)

            # Real API Call
            vlm_extraction, cost = self._call_vlm_api(b64_img)

            # Enforce cumulative budget
            self.budget_guard.check_and_add(cost)

            # Translate VLM output → internal schema
            for text_section in vlm_extraction.text_sections:
                block = TextBlock(
                    page_number=page_num,
                    bbox=None,
                    text=text_section.content,
                    content_hash="",
                )
                block.content_hash = block.generate_hash(text_section.content)
                blocks.append(block)

            for vlm_table in vlm_extraction.tables:
                rows = [[cell.value for cell in row.cells] for row in vlm_table.rows]
                block = TableBlock(
                    page_number=page_num,
                    bbox=None,
                    headers=vlm_table.headers,
                    rows=rows,
                    content_hash="",
                )
                block.content_hash = block.generate_hash(
                    str(vlm_table.headers) + str(rows)
                )
                blocks.append(block)

            # Structural validation before returning
            if not blocks:
                warnings.append("VLM returned empty extraction for this page")

            confidence = self._calculate_confidence(blocks, warnings)

            return ExtractionResult(
                page_number=page_num,
                blocks=blocks,
                confidence_score=confidence,
                strategy_used="vision_vlm",
                cost_estimate=cost,
                processing_time=time.time() - start_time,
                warnings=warnings,
            )

        except BudgetExceededError as e:
            warnings.append(str(e))
            return ExtractionResult(
                page_number=page_num,
                blocks=[],
                confidence_score=0.0,
                strategy_used="vision_vlm",
                cost_estimate=0.0,
                processing_time=time.time() - start_time,
                warnings=warnings,
            )
        except Exception as e:
            warnings.append(f"VLM API Error: {str(e)}")
            logger.error(f"VisionExtractor page {page_num} failed: {e}")
            return ExtractionResult(
                page_number=page_num,
                blocks=[],
                confidence_score=0.0,
                strategy_used="vision_vlm",
                cost_estimate=0.0,
                processing_time=time.time() - start_time,
                warnings=warnings,
            )
