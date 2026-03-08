import json, sys, logging
from pathlib import Path
from src.models.profile import DocumentProfile
from src.agents.router import ExtractionRouter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("run_single")

EXTRACTIONS_DIR = Path(".refinery/extractions")
DATA_DIR = Path("data/data")

def main(profile_name):
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    router = ExtractionRouter()
    
    if not profile_name.endswith(".json"):
        profile_name += ".json"
    profile_path = Path(".refinery/profiles") / profile_name
    with open(profile_path) as f:
        profile = DocumentProfile(**json.load(f))
        
    pdf_path = DATA_DIR / profile.file
    logger.info(f"Extracting: {profile.file}")
    doc = router.extract_document(str(pdf_path), profile)
    
    out_path = EXTRACTIONS_DIR / f"{profile.document_id}.json"
    with open(out_path, "w") as f:
        json.dump(doc.model_dump(), f, indent=2, default=str)
    
    logger.info(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    main(sys.argv[1])
