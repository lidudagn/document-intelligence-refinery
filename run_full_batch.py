import subprocess
import os
import glob
import json

# The 4 primary documents from the rubric + 8 additional from the corpus
# to reach the required minimum of 12 (3 per class).
docs = [
    # Class A: Financial Reports (native digital)
    "data/data/CBE ANNUAL REPORT 2023-24.pdf",          # Primary
    "data/data/CBE Annual Report 2018-19.pdf",
    "data/data/2021_Audited_Financial_Statement_Report.pdf",
    
    # Class B: Scanned Government/Legal (image-based)
    "data/data/Audit Report - 2023.pdf",                 # Primary
    "data/data/2013-E.C-Audit-finding-information.pdf",
    "data/data/2013-E.C-Procurement-information.pdf",
    
    # Class C: Technical Assessment / Mixed Layout
    "data/data/fta_performance_survey_final_report_2022.pdf",  # Primary
    "data/data/20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf",
    "data/data/Company_Profile_2024_25.pdf",
    
    # Class D: Structured / Numerical / Table-heavy
    "data/data/tax_expenditure_ethiopia_2021_22.pdf",    # Primary
    "data/data/Consumer Price Index August 2025.pdf",
    "data/data/Consumer Price Index September 2025.pdf",
]

print("=== STARTING FULL PIPELINE BATCH ===")
print(f"Processing {len(docs)} documents across 4 classes\n")

# 1. Triage all
print("--- Phase 1: Triage ---")
for doc in docs:
    if os.path.exists(doc):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        subprocess.run(["python3", "src/agents/triage.py", doc], env=env)
    else:
        print(f"WARNING: Missing doc: {doc}")

# 2. Extract all
print("\n--- Phase 2: Extraction ---")
subprocess.run(["python3", "run_extraction.py"])

# 3. Chunk + Index + QA all completed extractions
print("\n--- Phase 3 & 4: Semantic Chunking, Indexing, Vector Store, FactTable, and QA ---")
extracted_files = sorted(glob.glob(".refinery/extractions/*.json"))

for i, extraction_file in enumerate(extracted_files):
    base = os.path.basename(extraction_file)
    if "_pages" in base or "_ldu" in base:
        continue

    print(f"\n[{i+1}/{len(extracted_files)}] Processing {base[:40]}...")
    
    # Phase 3: Chunking & Indexing (saves LDUs, PageIndex, ingests to ChromaDB)
    try:
        subprocess.run(["python3", "run_indexer.py", "--json_path", extraction_file], check=True)
        print(f"  ✓ Phase 3 (Chunk + Index + VectorStore) complete")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Phase 3 Failed: {e}")
        continue
        
    # Phase 4: QA Generation (batch mode generates 3 Q&A pairs)
    try:
        subprocess.run(["python3", "run_query_agent.py", "--json_path", extraction_file, "--batch"], check=True)
        print(f"  ✓ Phase 4 (QA Batch) complete")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Phase 4 Failed: {e}")
        continue

print("\n=== BATCH PIPELINE COMPLETE ===")
