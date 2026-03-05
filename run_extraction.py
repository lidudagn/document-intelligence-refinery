"""
run_extraction.py — Run the Phase 2 Extraction Engine on the 4 representative documents.
Reads profiles from .refinery/profiles/, runs ExtractionRouter, and saves
ExtractedDocument JSON + ledger entries.
"""
import json
import sys
import logging
from pathlib import Path

from src.models.profile import DocumentProfile
from src.agents.router import ExtractionRouter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("run_extraction")

PROFILES_DIR = Path(".refinery/profiles")
EXTRACTIONS_DIR = Path(".refinery/extractions")
DATA_DIR = Path("data/data")


def main():
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)

    profiles = sorted(PROFILES_DIR.glob("*.json"))
    if not profiles:
        logger.error("No profiles found in %s", PROFILES_DIR)
        sys.exit(1)

    router = ExtractionRouter()

    for profile_path in profiles:
        with open(profile_path) as f:
            profile = DocumentProfile(**json.load(f))

        pdf_path = DATA_DIR / profile.file
        if not pdf_path.exists():
            logger.warning("PDF not found: %s — skipping", pdf_path)
            continue

        logger.info("=" * 60)
        logger.info("Extracting: %s", profile.file)
        logger.info("  origin=%s  layout=%s  cost=%s", profile.origin_type, profile.layout_complexity, profile.extraction_cost)

        try:
            doc = router.extract_document(str(pdf_path), profile)

            # Save extraction output
            out_path = EXTRACTIONS_DIR / f"{profile.document_id}.json"
            with open(out_path, "w") as f:
                json.dump(doc.model_dump(), f, indent=2, default=str)

            logger.info("  ✅ Saved to %s", out_path)
            logger.info("  pages=%d  confidence=%.3f", doc.total_pages, doc.overall_confidence)

            # Summary stats
            total_text_blocks = sum(
                1 for p in doc.pages for b in p.blocks if b.block_type == "text"
            )
            total_table_blocks = sum(
                1 for p in doc.pages for b in p.blocks if b.block_type == "table"
            )
            strategies_used = set(p.strategy_used for p in doc.pages)
            logger.info("  text_blocks=%d  table_blocks=%d  strategies=%s",
                        total_text_blocks, total_table_blocks, strategies_used)

        except Exception as e:
            logger.error("  ❌ Failed: %s", e)
            continue

    logger.info("=" * 60)
    logger.info("Done. Check .refinery/extractions/ and .refinery/extraction_ledger.jsonl")


if __name__ == "__main__":
    main()
