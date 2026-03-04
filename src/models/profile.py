from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict

class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"

class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"

class DomainHint(str, Enum):
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    GENERAL = "general"

class ExtractionCost(str, Enum):
    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"
    NEEDS_LAYOUT_MODEL = "needs_layout_model"
    NEEDS_VISION_MODEL = "needs_vision_model"
    
class DocumentProfile(BaseModel):
    document_id: str
    num_pages: int
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language: str
    domain_hint: DomainHint
    extraction_cost: ExtractionCost  # Note: This is a recommendation/escalation signal, not absolute final
    confidence_scores: Dict[str, float] = Field(..., description="Dict tracking origin_conf, layout_conf, lang_conf (0.0 to 1.0)")
    profiling_warnings: list[str] = Field(default_factory=list, description="Diagnostic array capturing edge cases (e.g., 'Inconsistent origin signals')")
    raw_metrics: Optional[Dict] = Field(default_factory=dict, description="Aggregated per-document heuristics (avg_char_density, std_dev_char_density, std_dev_column_count, etc.)")
