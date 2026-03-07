import time
from typing import List, Dict, Any
import pdfplumber

from src.strategies.base import BaseExtractor, ExtractionResult
from src.models.extracted_document import TextBlock, BoundingBox, BlockElement
from src.models.page_metrics import PageMetrics

class FastTextExtractor(BaseExtractor):
    """
    Strategy A: Low Cost
    Uses pdfplumber to quickly extract text strings and their bounding boxes.
    Applies a specific Confidence Scoring Formula to determine if escalation is needed.
    """
    
    def extract_page(self, pdf_path: str, page_num: int) -> ExtractionResult:
        start_time = time.time()
        blocks: List[BlockElement] = []
        
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            
            # Simple text extraction. Rather than just getting the page text,
            # we want to try and get structured blocks if possible, but pdfplumber usually
            # gives words or chars. A common approach is extract_words() and group them into lines or paragraphs.
            # Here we will do simple line clustering based on top coordinates.
            words = page.extract_words(keep_blank_chars=True)
            
            # Very rudimentary line forming
            current_line = []
            lines = []
            
            # Sort words top-to-bottom, left-to-right
            words = sorted(words, key=lambda w: (round(w['top'] / 5) * 5, w['x0']))
            
            for w in words:
                if not current_line:
                    current_line.append(w)
                else:
                    # Same line?
                    last_w = current_line[-1]
                    if abs(w['top'] - last_w['top']) < 3: # 3 pts tolerance
                        current_line.append(w)
                    else:
                        lines.append(current_line)
                        current_line = [w]
            if current_line:
                lines.append(current_line)
                
            # Convert lines to TextBlocks
            for line_words in lines:
                text = " ".join([w['text'] for w in line_words]).strip()
                if not text: continue
                
                x0 = min(w['x0'] for w in line_words)
                top = min(w['top'] for w in line_words)
                x1 = max(w['x1'] for w in line_words)
                bottom = max(w['bottom'] for w in line_words)
                
                block = TextBlock(
                    page_number=page_num + 1,
                    bbox=BoundingBox(x0=x0, top=top, x1=x1, bottom=bottom),
                    text=text,
                    content_hash="" # Placeholder, computed next
                )
                block.content_hash = block.generate_hash(text)
                blocks.append(block)
                
            # Confidence Scoring Metric
            # Here I will reuse the TriageAgent's PageMetrics logic loosely to get char density, etc.
            # However this extractor just needs those values to generate the final 0.0-1.0 confidence score.
            text_len = len(page.extract_text() or "")
            page_area = (page.width * page.height) if page.width and page.height else 1.0
            
            char_density = text_len / page_area
            
            img_area = sum((img['width'] * img['height']) for img in page.images)
            image_to_page_ratio = img_area / page_area
            
            # Formulas per requirements:
            # 0.40 * norm(char_density) + 0.40 * norm(char_count) + 0.20 * Penalty
            
            # Normalization bounds (empirical)
            norm_density = min(1.0, char_density / 0.02) # 0.02 is usually dense
            norm_count = min(1.0, text_len / 2000.0)    # 2000 chars is a very full page
            
            # Penalty for high image ratio: 1.0 if no images, decreases as images dominate
            img_score = max(0.0, 1.0 - (image_to_page_ratio * 1.5)) 
            
            confidence = (0.40 * norm_density) + (0.40 * norm_count) + (0.20 * img_score)
            
            # Post validation
            if text_len < 50 and image_to_page_ratio > 0.3:
                # Definitely a scanned page masquerading as digital
                confidence = min(0.3, confidence)
                
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            page_number=page_num + 1,
            blocks=blocks,
            confidence_score=float(confidence),
            strategy_used="fast_text",
            cost_estimate=0.0001, # Arbitrary small tracking cost for CPU compute
            processing_time=processing_time
        )
