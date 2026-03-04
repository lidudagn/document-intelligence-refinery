from pydantic import BaseModel
from typing import Optional

class PageMetrics(BaseModel):
    page_number: int
    char_density: float
    image_area_ratio: float
    table_count: int
    column_count: int
    whitespace_ratio: float
    has_text_layer: bool
    text_sample: Optional[str] = None
