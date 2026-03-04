import os
import json
from pathlib import Path
from src.agents.triage import TriageAgent

def main():
    corpus_dir = Path("data/data")
    if not corpus_dir.exists():
        print(f"Error: {corpus_dir} not found.")
        return

    agent = TriageAgent()
    
    pdfs = list(corpus_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in corpus directory.")
        return
        
    for pdf_path in pdfs:
        try:
            print(f"\n--- Processing {pdf_path.name} ---")
            profile = agent.profile_document(str(pdf_path))
            print(f"Origin: {profile.origin_type.value} ({profile.confidence_scores['origin_conf']:.2f})")
            print(f"Layout: {profile.layout_complexity.value} ({profile.confidence_scores['layout_conf']:.2f})")
            print(f"Domain: {profile.domain_hint.value}")
            print(f"Cost Routing: {profile.extraction_cost.value}")
            
            if profile.profiling_warnings:
                print("Warnings:", profile.profiling_warnings)
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

if __name__ == "__main__":
    main()
