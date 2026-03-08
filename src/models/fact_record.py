from pydantic import BaseModel, Field
from typing import Optional


class FactRecord(BaseModel):
    """
    A single extracted key-value fact from a document table.
    Rich dimensional schema: metric/entity/period enables SQL queries like
    WHERE entity='CBE' AND period='FY2023' AND metric='Revenue'.
    """
    fact_id: str
    document_id: str
    metric: str = Field(description="The measured quantity, e.g. 'Revenue', 'Net Income', 'Tax Expenditure'")
    entity: Optional[str] = Field(default=None, description="The org/entity, e.g. 'CBE', 'DBE', 'Ministry of Finance'")
    period: Optional[str] = Field(default=None, description="Time period, e.g. 'FY2023', 'Q3 2024', '2018/19-2020/21'")
    value: str = Field(description="Raw string value, e.g. '$4.2B', '12.5%'")
    unit: str = Field(default="", description="Unit: 'USD', 'ETB', 'percentage', 'count'")
    numeric_value: Optional[float] = Field(default=None, description="Parsed float for SQL comparisons")
    page_number: int
    bbox: Optional[list[float]] = Field(default=None, description="Bounding box [x0, top, x1, bottom]")
    content_hash: str = Field(description="SHA-256 hash from the source LDU")
    section: Optional[str] = Field(default=None, description="Parent section header")
    confidence: float = Field(default=1.0, description="Extraction confidence 0.0-1.0")
