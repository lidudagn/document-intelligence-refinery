import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional
import yaml
import pdfplumber
from tqdm import tqdm

from src.models import DocumentProfile, PageMetrics

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
                
                # TODO: Extract page metrics here
                
                # Memory management: flush cache for this page
                page.flush_cache()
                
            # Document-level cache flush
            pdf.flush()
                
        # TODO: Final aggregation and classification logic here...
        raise NotImplementedError("Classification mapping logic not yet implemented.")
