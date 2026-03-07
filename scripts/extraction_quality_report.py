import os
import json
import glob
from collections import defaultdict
import numpy as np

EXTRACTIONS_DIR = ".refinery/extractions"
REPORT_PATH = "EXTRACTION_QUALITY_ANALYSIS.md"

def generate_report():
    if not os.path.exists(EXTRACTIONS_DIR):
        print(f"Directory {EXTRACTIONS_DIR} not found.")
        return

    json_files = glob.glob(os.path.join(EXTRACTIONS_DIR, "*.json"))
    
    total_docs = len(json_files)
    total_tables = 0
    tables_per_doc = []
    
    all_rows = []
    all_cols = []
    confidences = []
    
    docs_with_tables = 0

    for fpath in json_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        doc_table_count = 0
        
        # Get overall doc confidence if present
        profile = data.get("profile", {})
        conf = profile.get("extraction_confidence", None)
        if conf is not None:
            confidences.append(float(conf))
            
        pages = data.get("pages", [])
        for page in pages:
            blocks = page.get("blocks", [])
            for block in blocks:
                if block.get("block_type") == "table":
                    doc_table_count += 1
                    total_tables += 1
                    
                    rows = block.get("rows", [])
                    headers = block.get("headers", [])
                    
                    num_rows = len(rows)
                    
                    # Columns is max length of rows or headers
                    num_cols = len(headers) if headers else 0
                    if rows:
                        num_cols = max(num_cols, max(len(r) for r in rows))
                        
                    all_rows.append(num_rows)
                    all_cols.append(num_cols)
                    
        tables_per_doc.append(doc_table_count)
        if doc_table_count > 0:
            docs_with_tables += 1

    avg_tables = np.mean(tables_per_doc) if tables_per_doc else 0
    avg_rows = np.mean(all_rows) if all_rows else 0
    avg_cols = np.mean(all_cols) if all_cols else 0
    
    avg_conf = np.mean(confidences) if confidences else 0
    
    report = f"""# Table Extraction Quality Report

## Overview
- **Total Documents Analyzed:** {total_docs}
- **Documents containing Tables:** {docs_with_tables} ({docs_with_tables/max(1, total_docs)*100:.1f}%)
- **Total Tables Extracted:** {total_tables}
- **Average Tables per Document:** {avg_tables:.2f}

## Table Dimensions
- **Average Rows per Table:** {avg_rows:.2f}
- **Average Columns per Table:** {avg_cols:.2f}

## Confidence Metrics
- **Average Document Extraction Confidence:** {avg_conf:.3f}
"""

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"Report generated at {REPORT_PATH}")

if __name__ == "__main__":
    generate_report()
