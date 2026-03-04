#!/usr/bin/env python3
"""
Phase 0 — Docling Corpus Analysis
===================================
Runs Docling DocumentConverter on the same 4 representative document classes
as analyze_pdfplumber.py. Computes matching metrics for direct comparison:
  - char_density, bbox_coverage, whitespace_ratio (same definitions as pdfplumber)
  - Docling-specific: sections, tables, figures detected
  - Sample table extraction (structured JSON for comparison)
  - Wall-clock timing + memory usage per document
  - Failure modes documented

Output: JSON results to stdout + scripts/docling_results.json
        Markdown exports to scripts/docling_output/
"""

import json
import os
import sys
import time
import traceback
import resource
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    print("⚠ Docling not installed. Run: pip install docling", file=sys.stderr)

# ── Document Classes (same as pdfplumber script) ─────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "docling_output"

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


def get_memory_mb():
    """Get current process memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(usage.ru_maxrss / 1024, 1)  # Linux: KB -> MB


def compute_docling_metrics(result, doc_info):
    """
    Compute metrics from Docling's conversion result.
    Uses the same definitions as pdfplumber for char_density,
    bbox_coverage, and whitespace_ratio to enable direct comparison.
    """
    doc = result.document
    failure_modes = []

    # ── Export markdown ──
    try:
        markdown_text = doc.export_to_markdown()
    except Exception as e:
        markdown_text = ""
        failure_modes.append(f"Markdown export failed: {e}")

    total_chars = len(markdown_text)

    # ── Count structural elements ──
    tables_found = 0
    figures_found = 0
    sections_found = 0
    sample_table = None

    try:
        for item, _level in doc.iterate_items():
            item_type = type(item).__name__
            if "Table" in item_type:
                tables_found += 1
                # Capture first table for concrete comparison
                if sample_table is None:
                    try:
                        table_data = item.export_to_dataframe() if hasattr(item, 'export_to_dataframe') else None
                        if table_data is not None:
                            sample_table = {
                                "headers": list(table_data.columns),
                                "rows": table_data.head(5).values.tolist(),
                                "total_rows": len(table_data),
                            }
                        else:
                            # Fallback: try to get text content
                            sample_table = {
                                "text": str(item)[:500],
                                "note": "DataFrame export not available",
                            }
                    except Exception as e:
                        sample_table = {"error": str(e)}
            elif "Figure" in item_type or "Picture" in item_type:
                figures_found += 1
            elif "Section" in item_type or "Heading" in item_type:
                sections_found += 1
    except Exception as e:
        failure_modes.append(f"Element iteration failed: {e}")

    # ── Per-page metrics (matching pdfplumber definitions) ──
    # Docling provides page-level information through its document model.
    # We approximate the same metrics using available element bboxes.
    page_metrics = []
    page_data = {}

    try:
        for item, _level in doc.iterate_items():
            # Get page references and bounding boxes from items
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    page_no = prov.page_no if hasattr(prov, 'page_no') else 0
                    if page_no not in page_data:
                        page_data[page_no] = {
                            "chars": 0,
                            "bbox_area": 0.0,
                            "page_area": 0.0,
                        }

                    # Count characters from this element's text
                    text = ""
                    if hasattr(item, 'text'):
                        text = item.text or ""
                    page_data[page_no]["chars"] += len(text)

                    # Accumulate bbox area from provenance
                    if hasattr(prov, 'bbox'):
                        bbox = prov.bbox
                        if hasattr(bbox, 'l') and hasattr(bbox, 'r'):
                            # Docling BoundingBox format
                            bw = abs(bbox.r - bbox.l)
                            bh = abs(bbox.b - bbox.t)
                            page_data[page_no]["bbox_area"] += bw * bh
                        elif hasattr(bbox, 'x0'):
                            bw = abs(bbox.x1 - bbox.x0)
                            bh = abs(bbox.y1 - bbox.y0)
                            page_data[page_no]["bbox_area"] += bw * bh

                    # Try to get page dimensions
                    if hasattr(prov, 'page_size') or hasattr(prov, 'size'):
                        size = getattr(prov, 'page_size', None) or getattr(prov, 'size', None)
                        if size and hasattr(size, 'width'):
                            page_data[page_no]["page_area"] = size.width * size.height
    except Exception as e:
        failure_modes.append(f"Page-level metric extraction failed: {e}")

    # Default page area from standard letter size if not detected
    DEFAULT_PAGE_AREA = 612 * 792  # US Letter in points

    for page_no in sorted(page_data.keys()):
        pd = page_data[page_no]
        page_area = pd["page_area"] if pd["page_area"] > 0 else DEFAULT_PAGE_AREA
        chars = pd["chars"]
        bbox_area = pd["bbox_area"]

        char_density = chars / page_area if page_area > 0 else 0
        bbox_coverage = min(bbox_area / page_area, 1.0) if page_area > 0 else 0
        whitespace_ratio = 1.0 - bbox_coverage

        page_metrics.append({
            "page_number": page_no,
            "total_chars": chars,
            "char_density": round(char_density, 8),
            "bbox_coverage": round(bbox_coverage, 6),
            "whitespace_ratio": round(whitespace_ratio, 6),
        })

    # ── Aggregate metrics ──
    if page_metrics:
        char_densities = [p["char_density"] for p in page_metrics]
        bbox_coverages = [p["bbox_coverage"] for p in page_metrics]
        whitespace_ratios = [p["whitespace_ratio"] for p in page_metrics]

        aggregate_metrics = {
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
        }
    else:
        aggregate_metrics = {
            "char_density": {"min": 0, "max": 0, "mean": 0},
            "bbox_coverage": {"min": 0, "max": 0, "mean": 0},
            "whitespace_ratio": {"min": 1, "max": 1, "mean": 1},
        }
        if total_chars == 0:
            failure_modes.append("Zero characters extracted — document may be scanned with no OCR")

    return {
        "total_chars": total_chars,
        "tables_found": tables_found,
        "figures_found": figures_found,
        "sections_found": sections_found,
        "sample_table": sample_table,
        "aggregate_metrics": aggregate_metrics,
        "page_metrics": page_metrics,
        "failure_modes": failure_modes,
        "markdown_length": len(markdown_text),
    }, markdown_text


def analyze_document(converter, doc_class, doc_info):
    """Analyze a single document with Docling."""
    filepath = DATA_DIR / doc_info["file"]
    if not filepath.exists():
        print(f"  ⚠ File not found: {filepath}", file=sys.stderr)
        return None

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {doc_class}: {doc_info['name']}", file=sys.stderr)
    print(f"  File: {doc_info['file']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    mem_before = get_memory_mb()
    start_time = time.time()
    failure_modes = []

    try:
        result = converter.convert(str(filepath), page_range=(1, 5))
        elapsed = time.time() - start_time
        mem_after = get_memory_mb()

        metrics, markdown = compute_docling_metrics(result, doc_info)
        failure_modes.extend(metrics["failure_modes"])

        # Save markdown export
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        md_path = OUTPUT_DIR / f"{doc_class}_{Path(doc_info['file']).stem}.md"
        with open(md_path, "w") as f:
            f.write(markdown)
        print(f"  ✓ Markdown saved to {md_path.name}", file=sys.stderr)

    except Exception as e:
        elapsed = time.time() - start_time
        mem_after = get_memory_mb()
        failure_modes.append(f"Conversion failed: {str(e)}")
        traceback.print_exc(file=sys.stderr)
        metrics = {
            "total_chars": 0,
            "tables_found": 0,
            "figures_found": 0,
            "sections_found": 0,
            "sample_table": None,
            "aggregate_metrics": {
                "char_density": {"min": 0, "max": 0, "mean": 0},
                "bbox_coverage": {"min": 0, "max": 0, "mean": 0},
                "whitespace_ratio": {"min": 1, "max": 1, "mean": 1},
            },
            "page_metrics": [],
            "failure_modes": failure_modes,
            "markdown_length": 0,
        }

    mem_delta = mem_after - mem_before

    print(f"  Time: {elapsed:.1f}s | Memory: +{mem_delta:.1f}MB "
          f"(peak {mem_after:.1f}MB)", file=sys.stderr)
    print(f"  Chars: {metrics['total_chars']} | Tables: {metrics['tables_found']} | "
          f"Sections: {metrics['sections_found']} | Figures: {metrics['figures_found']}",
          file=sys.stderr)

    if failure_modes:
        print(f"  ⚠ Failures: {failure_modes}", file=sys.stderr)

    return {
        "document_class": doc_class,
        "document_name": doc_info["name"],
        "file": doc_info["file"],
        "expected_origin": doc_info["origin"],
        "processing_time_seconds": round(elapsed, 2),
        "memory_delta_mb": round(mem_delta, 1),
        "peak_memory_mb": round(mem_after, 1),
        "total_chars": metrics["total_chars"],
        "tables_found": metrics["tables_found"],
        "figures_found": metrics["figures_found"],
        "sections_found": metrics["sections_found"],
        "sample_table": metrics["sample_table"],
        "char_density": metrics["aggregate_metrics"]["char_density"],
        "bbox_coverage": metrics["aggregate_metrics"]["bbox_coverage"],
        "whitespace_ratio": metrics["aggregate_metrics"]["whitespace_ratio"],
        "page_metrics": metrics["page_metrics"],
        "failure_modes": failure_modes,
        "markdown_length": metrics["markdown_length"],
    }


def main():
    if not HAS_DOCLING:
        print("Error: Docling is required. Install with: pip install docling",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 60, file=sys.stderr)
    print("  Docling Corpus Analysis — Phase 0", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # ── Initialize converter ──
    print("\n  Initializing DocumentConverter...", file=sys.stderr)
    init_start = time.time()

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for scanned docs
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            }
        )
        init_time = time.time() - init_start
        print(f"  ✓ Converter initialized in {init_time:.1f}s", file=sys.stderr)
    except Exception as e:
        print(f"  ✗ Converter init failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # ── Analyze each document ──
    results = {}
    docs_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(DOCUMENTS.keys())
    for doc_class in docs_to_run:
        if doc_class in DOCUMENTS:
            results[doc_class] = analyze_document(converter, doc_class, DOCUMENTS[doc_class])

    # ── Summary comparison ──
    print(f"\n\n{'='*60}", file=sys.stderr)
    print("  Summary: Docling Analysis Results", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  {'Class':<10} {'Chars':>8} {'Tables':>7} {'Sections':>9} "
          f"{'Time':>6} {'Memory':>8}", file=sys.stderr)
    print(f"  {'-'*50}", file=sys.stderr)

    for dc, data in results.items():
        if data:
            print(f"  {dc:<10} {data['total_chars']:>8} {data['tables_found']:>7} "
                  f"{data['sections_found']:>9} {data['processing_time_seconds']:>5.1f}s "
                  f"{data['memory_delta_mb']:>+6.1f}MB", file=sys.stderr)

    # ── Output ──
    output = {
        "analysis_tool": "docling",
        "converter_init_time_seconds": round(init_time, 2),
        "document_results": {
            k: {key: val for key, val in v.items() if key != "page_metrics"}
            for k, v in results.items() if v is not None
        },
        "page_level_data": {
            k: v["page_metrics"]
            for k, v in results.items() if v is not None
        },
    }

    suffix = "_" + sys.argv[1] if len(sys.argv) > 1 else ""
    output_path = Path(__file__).resolve().parent / f"docling_results{suffix}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  ✓ Results saved to {output_path}", file=sys.stderr)

    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
