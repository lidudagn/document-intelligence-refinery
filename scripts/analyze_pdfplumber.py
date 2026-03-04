#!/usr/bin/env python3
"""
Phase 0 — pdfplumber Corpus Analysis
=====================================
Computes per-page metrics for 4 representative document classes:
  - char_density, bbox_coverage, whitespace_ratio
  - image_area_ratio, table_count, font_inventory, scanned_page_ratio
  - Threshold sensitivity sweep for scanned detection
  - Wall-clock timing per document

Output: JSON results to stdout + scripts/pdfplumber_results.json
"""

import json
import os
import sys
import time
from pathlib import Path

import pdfplumber

# ── Document Classes ──────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data"

DOCUMENTS = {
    "Class_A": {
        "name": "CBE Annual Report 2023-24",
        "file": "CBE ANNUAL REPORT 2023-24.pdf",
        "origin": "native_digital",
        "description": "Multi-column, tables, financial statements",
    },
    "Class_B": {
        "name": "DBE Audit Report 2023",
        "file": "Audit Report - 2023.pdf",
        "origin": "scanned_image",
        "description": "Scanned image, no text layer",
    },
    "Class_C": {
        "name": "FTA Performance Survey 2022",
        "file": "fta_performance_survey_final_report_2022.pdf",
        "origin": "native_digital",
        "description": "Mixed tables + narrative, high font diversity",
    },
    "Class_D": {
        "name": "Tax Expenditure Report 2021-22",
        "file": "tax_expenditure_ethiopia_2021_22.pdf",
        "origin": "native_digital",
        "description": "Table-heavy, numerical fiscal data",
    },
}


def compute_page_metrics(page):
    """Compute all metrics for a single page."""
    width = page.width
    height = page.height
    page_area = width * height

    # ── Character density ──
    chars = page.chars or []
    total_chars = len(chars)
    char_density = total_chars / page_area if page_area > 0 else 0

    # ── Bbox coverage (union of char bboxes / page area) ──
    # We approximate union area by summing individual char bbox areas.
    # For overlapping chars this slightly overestimates, but the relative
    # difference between scanned (0) and native (>0) is what matters.
    char_bbox_area = 0.0
    for c in chars:
        cw = abs(c["x1"] - c["x0"])
        ch = abs(c["bottom"] - c["top"])
        char_bbox_area += cw * ch
    bbox_coverage = min(char_bbox_area / page_area, 1.0) if page_area > 0 else 0

    # ── Whitespace ratio (1 - bbox_coverage) ──
    whitespace_ratio = 1.0 - bbox_coverage

    # ── Image area ratio ──
    images = page.images or []
    image_area = 0.0
    for img in images:
        iw = abs(img["x1"] - img["x0"])
        ih = abs(img["bottom"] - img["top"])
        image_area += iw * ih
    image_area_ratio = min(image_area / page_area, 1.0) if page_area > 0 else 0

    # ── Table count ──
    try:
        tables = page.find_tables()
        table_count = len(tables)
    except Exception:
        table_count = 0

    # ── Font inventory ──
    fonts = set()
    for c in chars:
        font_name = c.get("fontname", "unknown")
        font_size = round(c.get("size", 0), 1)
        fonts.add((font_name, font_size))

    return {
        "page_area": round(page_area, 2),
        "total_chars": total_chars,
        "char_density": round(char_density, 8),
        "bbox_coverage": round(bbox_coverage, 6),
        "whitespace_ratio": round(whitespace_ratio, 6),
        "image_area_ratio": round(image_area_ratio, 6),
        "table_count": table_count,
        "font_count": len(fonts),
        "fonts": sorted([f"{name}@{size}" for name, size in fonts]),
    }


def analyze_document(doc_class, doc_info):
    """Analyze a single document and return full metrics."""
    filepath = DATA_DIR / doc_info["file"]
    if not filepath.exists():
        print(f"  ⚠ File not found: {filepath}", file=sys.stderr)
        return None

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {doc_class}: {doc_info['name']}", file=sys.stderr)
    print(f"  File: {doc_info['file']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    start_time = time.time()
    page_metrics = []
    all_fonts = set()

    with pdfplumber.open(filepath) as pdf:
        total_pages = len(pdf.pages)
        print(f"  Pages: {total_pages}", file=sys.stderr)

        for i, page in enumerate(pdf.pages):
            metrics = compute_page_metrics(page)
            metrics["page_number"] = i + 1
            page_metrics.append(metrics)

            # Collect fonts across pages
            for f in metrics["fonts"]:
                all_fonts.add(f)

            if (i + 1) % 20 == 0:
                print(f"    ... processed page {i+1}/{total_pages}", file=sys.stderr)

    elapsed = time.time() - start_time

    # ── Aggregate metrics ──
    char_densities = [p["char_density"] for p in page_metrics]
    bbox_coverages = [p["bbox_coverage"] for p in page_metrics]
    whitespace_ratios = [p["whitespace_ratio"] for p in page_metrics]
    image_ratios = [p["image_area_ratio"] for p in page_metrics]
    char_counts = [p["total_chars"] for p in page_metrics]
    table_counts = [p["table_count"] for p in page_metrics]

    # Scanned page detection: chars < 100 AND image_ratio > 0.50
    scanned_pages = [
        p["page_number"]
        for p in page_metrics
        if p["total_chars"] < 100 and p["image_area_ratio"] > 0.50
    ]
    scanned_page_ratio = len(scanned_pages) / total_pages if total_pages > 0 else 0

    aggregate = {
        "document_class": doc_class,
        "document_name": doc_info["name"],
        "file": doc_info["file"],
        "expected_origin": doc_info["origin"],
        "total_pages": total_pages,
        "processing_time_seconds": round(elapsed, 2),
        "total_chars": sum(char_counts),
        "total_tables": sum(table_counts),
        "unique_fonts": len(all_fonts),
        "scanned_pages": scanned_pages,
        "scanned_page_count": len(scanned_pages),
        "scanned_page_ratio": round(scanned_page_ratio, 4),
        "char_density": {
            "min": round(min(char_densities), 8),
            "max": round(max(char_densities), 8),
            "mean": round(sum(char_densities) / len(char_densities), 8),
        },
        "bbox_coverage": {
            "min": round(min(bbox_coverages), 6),
            "max": round(max(bbox_coverages), 6),
            "mean": round(sum(bbox_coverages) / len(bbox_coverages), 6),
        },
        "whitespace_ratio": {
            "min": round(min(whitespace_ratios), 6),
            "max": round(max(whitespace_ratios), 6),
            "mean": round(sum(whitespace_ratios) / len(whitespace_ratios), 6),
        },
        "image_area_ratio": {
            "min": round(min(image_ratios), 6),
            "max": round(max(image_ratios), 6),
            "mean": round(sum(image_ratios) / len(image_ratios), 6),
        },
    }

    print(f"  ✓ Done in {elapsed:.1f}s — {total_pages} pages, "
          f"{sum(char_counts)} chars, {sum(table_counts)} tables", file=sys.stderr)

    return {
        "aggregate": aggregate,
        "page_metrics": page_metrics,
    }


def threshold_sensitivity_sweep(results):
    """
    For char_density threshold: test at 5 values and count
    false positives (native pages classified as scanned) and
    false negatives (scanned pages missed).
    """
    thresholds = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    sweep_results = []

    for threshold in thresholds:
        false_scanned = {}  # native pages wrongly called scanned
        missed_scanned = {}  # scanned pages wrongly called native

        for doc_class, data in results.items():
            if data is None:
                continue
            is_scanned_doc = data["aggregate"]["expected_origin"] == "scanned_image"
            false_scanned_pages = []
            missed_scanned_pages = []

            for page in data["page_metrics"]:
                page_is_scanned = page["char_density"] < threshold
                if page_is_scanned and not is_scanned_doc:
                    # False positive: native page called scanned
                    false_scanned_pages.append(page["page_number"])
                elif not page_is_scanned and is_scanned_doc:
                    # False negative: scanned page called native
                    missed_scanned_pages.append(page["page_number"])

            if false_scanned_pages:
                false_scanned[doc_class] = false_scanned_pages
            if missed_scanned_pages:
                missed_scanned[doc_class] = missed_scanned_pages

        total_false = sum(len(v) for v in false_scanned.values())
        total_missed = sum(len(v) for v in missed_scanned.values())

        sweep_results.append({
            "threshold": threshold,
            "false_scanned_total": total_false,
            "false_scanned_detail": false_scanned,
            "missed_scanned_total": total_missed,
            "missed_scanned_detail": missed_scanned,
            "verdict": "✅ Clean" if total_false == 0 and total_missed == 0 else "⚠ Has errors",
        })

    return sweep_results


def extract_sample_table(doc_info, page_number=None):
    """
    Extract a sample table from a document to show pdfplumber's
    flat-text output for the concrete comparison section.
    If page_number is None, finds the first page with a table.
    """
    filepath = DATA_DIR / doc_info["file"]
    if not filepath.exists():
        return None

    with pdfplumber.open(filepath) as pdf:
        if page_number is not None:
            pages_to_check = [pdf.pages[page_number - 1]]
        else:
            pages_to_check = pdf.pages[:30]  # check first 30 pages

        for page in pages_to_check:
            try:
                tables = page.find_tables()
                if tables:
                    table = tables[0]
                    extracted = table.extract()
                    return {
                        "page_number": page.page_number,
                        "rows": extracted[:5],  # first 5 rows
                        "total_rows": len(extracted),
                        "flat_text": "\n".join(
                            "  |  ".join(str(cell or "") for cell in row)
                            for row in extracted[:5]
                        ),
                    }
            except Exception:
                continue
    return None


def main():
    print("=" * 60, file=sys.stderr)
    print("  pdfplumber Corpus Analysis — Phase 0", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results = {}
    for doc_class, doc_info in DOCUMENTS.items():
        results[doc_class] = analyze_document(doc_class, doc_info)

    # ── Threshold sensitivity sweep ──
    print("\n\n" + "=" * 60, file=sys.stderr)
    print("  Threshold Sensitivity Sweep (char_density)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    sweep = threshold_sensitivity_sweep(results)
    for entry in sweep:
        status = entry["verdict"]
        print(f"  threshold={entry['threshold']:.4f}  "
              f"false_scanned={entry['false_scanned_total']}  "
              f"missed_scanned={entry['missed_scanned_total']}  "
              f"{status}", file=sys.stderr)

    # ── Sample tables for concrete comparison ──
    print("\n\n" + "=" * 60, file=sys.stderr)
    print("  Sample Table Extraction (for comparison)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    sample_tables = {}
    for doc_class, doc_info in DOCUMENTS.items():
        table = extract_sample_table(doc_info)
        if table:
            sample_tables[doc_class] = table
            print(f"  {doc_class}: Table found on page {table['page_number']} "
                  f"({table['total_rows']} rows)", file=sys.stderr)
        else:
            print(f"  {doc_class}: No tables found", file=sys.stderr)

    # ── Assemble output ──
    output = {
        "analysis_tool": "pdfplumber",
        "document_results": {
            k: v["aggregate"] for k, v in results.items() if v is not None
        },
        "page_level_data": {
            k: v["page_metrics"] for k, v in results.items() if v is not None
        },
        "threshold_sensitivity": sweep,
        "sample_tables": sample_tables,
    }

    # Save to file
    output_path = Path(__file__).resolve().parent / "pdfplumber_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  ✓ Results saved to {output_path}", file=sys.stderr)

    # Also print to stdout
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
