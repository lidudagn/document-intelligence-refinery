import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Dict
import yaml
import pdfplumber
from tqdm import tqdm
import numpy as np
from langdetect import detect_langs, DetectorFactory

from src.models import DocumentProfile, PageMetrics
from src.models.profile import OriginType, LayoutComplexity, DomainHint, ExtractionCost

# Ensure deterministic language detection
DetectorFactory.seed = 0

class TriageAgent:
    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        self.config = self._load_config(config_path)
        self.cache_dir = Path(".refinery/profiles")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)["triage"]

    def _generate_document_id(self, pdf_path: str) -> str:
        """Computes a consistent document_id via SHA256 hash of filename + file contents."""
        path = Path(pdf_path)
        hasher = hashlib.sha256()
        hasher.update(path.name.encode('utf-8'))
        
        # Hash the first 1MB of the file to be efficient but unique enough
        with open(path, 'rb') as f:
            chunk = f.read(1024 * 1024)
            hasher.update(chunk)
            
        return hasher.hexdigest()

    def _get_pages_to_sample(self, num_pages: int) -> List[int]:
        """Determine which pages to sample based on YAML config, safely bounding checks."""
        first_n = self.config["page_sampling"]["first_n"]
        mid_n = self.config["page_sampling"]["mid_n"]
        last_n = self.config["page_sampling"]["last_n"]
        
        total_requested = first_n + mid_n + last_n
        
        if num_pages <= total_requested:
            return list(range(num_pages))
            
        sampled_pages = set()
        
        # First N
        for i in range(min(first_n, num_pages)):
            sampled_pages.add(i)
            
        # Last N
        for i in range(max(0, num_pages - last_n), num_pages):
            sampled_pages.add(i)
            
        # Mid N
        mid_start = max(0, (num_pages // 2) - (mid_n // 2))
        for i in range(mid_start, min(num_pages, mid_start + mid_n)):
            sampled_pages.add(i)
            
        return sorted(list(sampled_pages))

    def _extract_page_metrics(self, page: pdfplumber.page.Page, page_num: int) -> PageMetrics:
        """Extracts single-page heuristics."""
        page_area = page.width * page.height if page.width and page.height else 1.0
        
        # Characters and generic text metadata
        chars = page.chars
        char_count = len(chars)
        char_density = char_count / float(page_area)
        
        # Text block recovery for language parsing
        text_sample = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
        text_sample = text_sample.strip()
        has_text_layer = len(text_sample) > 50
        
        # Image area
        images = page.images
        img_area = sum(img["width"] * img["height"] for img in images) if images else 0
        image_area_ratio = img_area / float(page_area)
        
        # Tables
        tables = page.find_tables()
        table_count = len(tables) if tables else 0
        
        # Columns (X-coordinate clustering)
        column_count = 1
        words = page.extract_words()
        if words:
            # Sort words horizontally
            x_coords = sorted([(w["x0"], w["x1"]) for w in words], key=lambda x: x[0])
            clusters = []
            gap_tolerance = self.config["thresholds"]["column_gap_tolerance_pts"]
            
            for x0, x1 in x_coords:
                placed = False
                for cluster in clusters:
                    # If this word horizontal bound is within tolerance of existing column boundary, expand cluster
                    if max(0, x0 - cluster[1]) < gap_tolerance:
                        cluster[0] = min(cluster[0], x0)
                        cluster[1] = max(cluster[1], x1)
                        placed = True
                        break
                if not placed:
                    clusters.append([x0, x1])
                    
            if len(clusters) > 1:
                # Need to verify if the clusters aren't just one-off tables.
                # Only count distinct columns if they have robust height density (not just a header block)
                column_count = len(clusters)
        
        # Whitespace
        # Simplifying bounding box coverage as requested - character density is primary text representation
        # True blank pages will have nearly 1.0 whitespace and 0 character density
        whitespace_ratio = 1.0 if char_count == 0 else max(0.0, 1.0 - (char_count * 10 / page_area))

        return PageMetrics(
            page_number=page_num,
            char_density=char_density,
            image_area_ratio=image_area_ratio,
            table_count=table_count,
            column_count=column_count,
            whitespace_ratio=whitespace_ratio,
            has_text_layer=has_text_layer,
            text_sample=text_sample if has_text_layer else None
        )

    def _classify_origin(self, page_metrics: List[PageMetrics], is_acroform: bool) -> tuple[OriginType, float]:
        """Matrix evaluator for Origin Type and its respective statistical confidence."""
        if is_acroform:
            return OriginType.FORM_FILLABLE, 1.0
            
        native_thresh = self.config["thresholds"]["char_density_native_threshold"]
        
        # Determine S and N ratios
        scanned_pages = sum(1 for p in page_metrics if p.char_density < native_thresh and p.image_area_ratio > 0.5)
        native_pages = sum(1 for p in page_metrics if p.char_density >= native_thresh)
        
        total = len(page_metrics)
        if total == 0:
            return OriginType.NATIVE_DIGITAL, 0.0 # Blind fallback
            
        S = scanned_pages / total
        N = native_pages / total
        
        origin_conf = max(S, N)
        
        scanned_matrix_thresh = self.config["thresholds"]["scanned_page_ratio_threshold"]
        native_matrix_thresh = self.config["thresholds"]["native_page_ratio_threshold"]
        mixed_bound = self.config["thresholds"]["mixed_lower_bound"]
        
        if S >= scanned_matrix_thresh:
            return OriginType.SCANNED_IMAGE, origin_conf
        elif N >= native_matrix_thresh:
            return OriginType.NATIVE_DIGITAL, origin_conf
        elif mixed_bound <= S < scanned_matrix_thresh:
            return OriginType.MIXED, origin_conf
        else:
            return OriginType.NATIVE_DIGITAL, origin_conf

    def _classify_layout(self, page_metrics: List[PageMetrics]) -> tuple[LayoutComplexity, float]:
        """Determines layout using deterministic hierarchy to prevent unstable signals."""
        total = len(page_metrics)
        if total == 0:
            return LayoutComplexity.SINGLE_COLUMN, 0.0
            
        table_heavy_limit = self.config["thresholds"]["table_heavy_page_ratio"]
        
        # 1. Table heavy
        pages_with_tables = sum(1 for p in page_metrics if p.table_count > 0)
        table_ratio = pages_with_tables / total
        if table_ratio >= table_heavy_limit:
            return LayoutComplexity.TABLE_HEAVY, table_ratio
            
        # 2. Figure heavy (Simplified proxy via image ratio mapping on native pages)
        pages_with_figures = sum(1 for p in page_metrics if p.image_area_ratio > 0.3 and p.char_density >= 0.001)
        figure_ratio = pages_with_figures / total
        if figure_ratio >= 0.4:
            return LayoutComplexity.FIGURE_HEAVY, figure_ratio
            
        # 3. Multi Column
        pages_with_columns = sum(1 for p in page_metrics if p.column_count > 1)
        col_ratio = pages_with_columns / total
        if col_ratio >= 0.3:
            return LayoutComplexity.MULTI_COLUMN, col_ratio
            
        # 4. Fallback Single Column
        return LayoutComplexity.SINGLE_COLUMN, 1.0 - max(table_ratio, figure_ratio, col_ratio)

    def _detect_language(self, page_metrics: List[PageMetrics], origin_type: OriginType) -> tuple[str, float]:
        """Evaluates language, gracefully bypassing ML models if pure scanned."""
        if origin_type == OriginType.SCANNED_IMAGE:
            return "unknown", 0.0
            
        # Aggregate text samples from native pages
        full_text = " ".join([p.text_sample for p in page_metrics if p.text_sample])
        if len(full_text) < 50:
            return "unknown", 0.0
            
        try:
            detected = detect_langs(full_text[:5000])[0] # Only need first 5k chars for confidence
            return detected.lang, detected.prob
        except:
            return "unknown", 0.0

    def _detect_domain(self, page_metrics: List[PageMetrics]) -> tuple[DomainHint, float]:
        """Weighted scoring approach to domain determination."""
        full_text = " ".join([p.text_sample for p in page_metrics if p.text_sample]).lower()
        if not full_text:
            return DomainHint.GENERAL, 0.0
            
        # Domain dictionaries and associated weights
        domains = {
            DomainHint.FINANCIAL: {"revenue": 5, "fiscal": 5, "balance sheet": 5, "tax expenditure": 4, "income statement": 5, "audit": 3},
            DomainHint.LEGAL: {"whereas": 5, "hereby": 4, "jurisdiction": 5, "affidavit": 5, "plaintiff": 5, "statute": 3},
            DomainHint.TECHNICAL: {"architecture": 4, "protocol": 4, "latency": 5, "system design": 5, "api": 4, "bandwidth": 3},
            DomainHint.MEDICAL: {"patient": 5, "diagnosis": 5, "clinical": 4, "symptoms": 4, "treatment": 3, "syndrome": 5}
        }
        
        scores = {d: 0 for d in DomainHint if d != DomainHint.GENERAL}
        total_hits = 0
        
        for domain, keywords in domains.items():
            for kw, weight in keywords.items():
                hits = full_text.count(kw)
                scores[domain] += hits * weight
                total_hits += hits
                
        if total_hits == 0:
            return DomainHint.GENERAL, 0.0
            
        winning_domain = max(scores, key=scores.get)
        max_score = scores[winning_domain]
        
        if max_score < 10: # Minimum activation threshold
            return DomainHint.GENERAL, 0.0
            
        confidence = min(1.0, max_score / (total_hits * 5.0))
        return winning_domain, confidence

    def _route_extraction(self, origin: OriginType, o_conf: float, layout: LayoutComplexity, l_conf: float) -> ExtractionCost:
        """Confidence-Aware Routing Matrix."""
        routing_o_thresh = self.config["thresholds"]["routing_origin_threshold"]
        routing_l_thresh = self.config["thresholds"]["routing_layout_threshold"]
        
        if o_conf < routing_o_thresh:
            return ExtractionCost.NEEDS_LAYOUT_MODEL
        elif origin == OriginType.SCANNED_IMAGE:
            return ExtractionCost.NEEDS_VISION_MODEL
        elif layout in [LayoutComplexity.TABLE_HEAVY, LayoutComplexity.MULTI_COLUMN] and l_conf >= routing_l_thresh:
            return ExtractionCost.NEEDS_LAYOUT_MODEL
        else:
            return ExtractionCost.FAST_TEXT_SUFFICIENT

    def profile_document(self, pdf_path: str) -> DocumentProfile:
        doc_id = self._generate_document_id(pdf_path)
        cache_file = self.cache_dir / f"{doc_id}.json"
        
        if cache_file.exists():
            print(f"Loading cached profile for {doc_id}")
            with open(cache_file, "r") as f:
                return DocumentProfile.model_validate_json(f.read())
                
        print(f"Profiling new document: {Path(pdf_path).name}")
        
        page_metrics_list: List[PageMetrics] = []
        is_acroform = False
        
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            if pdf.doc.catalog.get("AcroForm"):
                is_acroform = True
                
            pages_to_sample = self._get_pages_to_sample(num_pages)
            
            for page_num in tqdm(pages_to_sample, desc="Extracting heuristics"):
                page = pdf.pages[page_num]
                metrics = self._extract_page_metrics(page, page_num)
                page_metrics_list.append(metrics)
                
                # Memory management: flush cache for this page
                page.flush_cache()
                
            # Document-level cache flush
            pdf.flush()
                
        # 1. Aggregation and Variance
        densities = [m.char_density for m in page_metrics_list]
        columns = [m.column_count for m in page_metrics_list]
        img_ratios = [m.image_area_ratio for m in page_metrics_list]
        
        raw_metrics = {
            "avg_char_density": float(np.mean(densities)) if densities else 0.0,
            "std_dev_char_density": float(np.std(densities)) if densities else 0.0,
            "min_char_density": float(np.min(densities)) if densities else 0.0,
            "avg_column_count": float(np.mean(columns)) if columns else 1.0,
            "std_dev_column_count": float(np.std(columns)) if columns else 0.0,
            "avg_image_ratio": float(np.mean(img_ratios)) if img_ratios else 0.0,
            "max_image_ratio": float(np.max(img_ratios)) if img_ratios else 0.0,
            "total_tables_detected": sum(m.table_count for m in page_metrics_list)
        }
        
        # 2. Heuristics Evaluation
        origin_type, origin_conf = self._classify_origin(page_metrics_list, is_acroform)
        layout_complexity, layout_conf = self._classify_layout(page_metrics_list)
        language, lang_conf = self._detect_language(page_metrics_list, origin_type)
        domain_hint, domain_conf = self._detect_domain(page_metrics_list)
        extraction_cost = self._route_extraction(origin_type, origin_conf, layout_complexity, layout_conf)
        
        # 3. Defensive Warnings
        warnings = []
        if origin_conf < 0.6:
            warnings.append("Inconsistent origin signals (high variance between native and scanned metrics).")
        if layout_conf < 0.5:
            warnings.append("Weak layout classification confidence.")
        if language == "unknown":
            warnings.append("Failed to detect language; insufficient textual stream.")
            
        profile = DocumentProfile(
            document_id=doc_id,
            num_pages=num_pages,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            language=language,
            domain_hint=domain_hint,
            extraction_cost=extraction_cost,
            confidence_scores={
                "origin_conf": origin_conf,
                "layout_conf": layout_conf,
                "lang_conf": lang_conf,
                "domain_conf": domain_conf
            },
            profiling_warnings=warnings,
            raw_metrics=raw_metrics
        )
        
        # 4. Cache and Return
        with open(cache_file, "w") as f:
            f.write(profile.model_dump_json(indent=2))
            
        return profile
