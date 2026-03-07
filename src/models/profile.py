from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List

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

class ExtractionCost(str, Enum):
    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"
    NEEDS_LAYOUT_MODEL = "needs_layout_model"
    NEEDS_VISION_MODEL = "needs_vision_model"
    
class DocumentProfile(BaseModel):
    document_id: str
    file: str = Field(..., description="Source PDF filename")
    num_pages: int
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language: str
    domain_hint: str = Field(description="Dynamic domain hint (e.g., 'financial', 'legal') sourced from config")
    extraction_cost: ExtractionCost  # Note: This is a recommendation/escalation signal, not absolute final
    confidence_scores: Dict[str, float] = Field(..., description="Dict tracking origin_conf, layout_conf, lang_conf (0.0 to 1.0)")
    profiling_warnings: list[str] = Field(default_factory=list, description="Diagnostic array capturing edge cases (e.g., 'Inconsistent origin signals')")
    raw_metrics: Optional[Dict] = Field(default_factory=dict, description="Aggregated per-document heuristics (avg_char_density, std_dev_char_density, std_dev_column_count, etc.)")

    @field_validator('num_pages')
    @classmethod
    def validate_num_pages(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('num_pages must be greater than 0')
        return v

    @field_validator('confidence_scores')
    @classmethod
    def validate_confidence_scores(cls, v: Dict[str, float]) -> Dict[str, float]:
        for key, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(f'Confidence score for {key} must be between 0.0 and 1.0, got {score}')
        return v

    @field_validator('file')
    @classmethod
    def validate_file(cls, v: str) -> str:
        if not v.lower().endswith('.pdf'):
            raise ValueError('file must end with .pdf')
        return v

    @field_validator('domain_hint')
    @classmethod
    def validate_domain_hint(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('domain_hint cannot be empty whitespace')
        return v

    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('language must be defined')
        return v
