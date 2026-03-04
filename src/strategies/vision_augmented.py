import time
import base64
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import fitz # PyMuPDF

from src.strategies.base import BaseExtractor, ExtractionResult
from src.models.extracted_document import BlockElement, TextBlock, TableBlock, ExtractedDocument

class BudgetExceededError(Exception):
    pass

class BudgetGuard:
    """Enforces absolute real-world dollar limits on LLM API calls per document."""
    
    def __init__(self, max_usd_per_doc: float = 0.10):
        self.max_usd = max_usd_per_doc
        self.current_spend = 0.0
        
    def add_cost(self, usd_cost: float):
        if self.current_spend + usd_cost > self.max_usd:
            raise BudgetExceededError(f"Extraction halted: cost ${self.current_spend + usd_cost:.4f} exceeds max budget ${self.max_usd:.2f}")
        self.current_spend += usd_cost

# --- VLM Prompt Templates & Schemas ---
# We enforce structured output using these intermediate Pydantic Schemas to ensure we can 
# cleanly map the LLM response back to our internal TextBlock/TableBlock models.

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
    text_sections: List[VLMTextSection] = Field(default_factory=list, description="All paragraphs and textual narrative on the page")
    tables: List[VLMTable] = Field(default_factory=list, description="Any tabular data found on the page")

VISION_PROMPT = """
You are an expert Document Intelligence Vision Language Model.
Your task is to extract all structural content from the provided page image exactly as it appears. 
Identify all tables and extract them with strict row and column alignment. 
Identify all continuous narrative paragraphs and extract them as text sections.
Do not hallucinate content. Ignore headers and footers that repeat across pages.
"""

class VisionExtractor(BaseExtractor):
    """
    Strategy C: High Cost
    Uses an LLM Multimodal completion API to parse complex, degraded, or pure-image pages.
    Tracks budget internally to prevent runaway costs.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        vision_cfg = self.config.get("extraction", {}).get("vision", {})
        max_usd = vision_cfg.get("max_usd_per_document", 0.10)
        self.budget_guard = BudgetGuard(max_usd_per_doc=max_usd)
        
    def _render_page_to_base64(self, pdf_path: str, page_num: int) -> str:
        """Render a specific PDF page as an image to pass to the VLM."""
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        # Render high quality image
        zoom = 2.0 
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")
        
    def _call_vlm_api(self, base64_image: str) -> tuple[VLMPageExtraction, float]:
        """
        Mock call to a structured VLM API (e.g. OpenAI structured outputs / OpenRouter).
        In a real deployment, this would use the `openai` client with `response_format=VLMPageExtraction`.
        Currently returns empty structures to satisfy types while testing.
        """
        # --- Real implementation placeholder ---
        # response = client.beta.chat.completions.parse(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": VISION_PROMPT},
        #         {"role": "user", "content": [
        #             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        #         ]}
        #     ],
        #     response_format=VLMPageExtraction
        # )
        # cost = (response.usage.prompt_tokens * 0.150 / 1e6) + (response.usage.completion_tokens * 0.600 / 1e6)
        # return response.choices[0].message.parsed, cost
        
        # Fake cost for testing: roughly 0.002 to 0.005 per page for short docs on 4o-mini
        simulated_cost = 0.003 
        
        return VLMPageExtraction(text_sections=[], tables=[]), simulated_cost

    def extract_page(self, pdf_path: str, page_num: int) -> ExtractionResult:
        start_time = time.time()
        blocks: List[BlockElement] = []
        warnings = []
        
        try:
            b64_img = self._render_page_to_base64(pdf_path, page_num)
            
            # API Call
            vlm_extraction, cost = self._call_vlm_api(b64_img)
            
            # Enforce Budget
            self.budget_guard.add_cost(cost)
            
            # Translate VLM Output to ExtractedDocument schemas
            for text_section in vlm_extraction.text_sections:
                block = TextBlock(
                    page_number=page_num,
                    bbox=None, # VLM structured output rarely provides accurate bounding boxes natively
                    text=text_section.content,
                    content_hash=""
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
                    content_hash=""
                )
                block.content_hash = block.generate_hash(str(vlm_table.headers) + str(rows))
                blocks.append(block)
                
            confidence = 0.95 # VLM output is usually structural ground truth, though hallucination is possible.
            
            return ExtractionResult(
                page_number=page_num,
                blocks=blocks,
                confidence_score=confidence,
                strategy_used="vision_vlm",
                cost_estimate=cost,
                processing_time=time.time() - start_time,
                warnings=warnings
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
                warnings=warnings
            )
        except Exception as e:
            warnings.append(f"VLM API Error: {str(e)}")
            return ExtractionResult(
                page_number=page_num,
                blocks=[],
                confidence_score=0.0,
                strategy_used="vision_vlm",
                cost_estimate=0.0,
                processing_time=time.time() - start_time,
                warnings=warnings
            )
