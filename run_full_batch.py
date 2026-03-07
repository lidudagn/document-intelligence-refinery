import subprocess
import os
import glob
import json

# We need 12 documents spanning 4 classes (Financial Native, Scanned, Mixed Tech, Structured Native).
docs = [
    # Class A: Financial Reports
    "data/data/2021_Audited_Financial_Statement_Report.pdf",
    "data/data/CBE ANNUAL REPORT 2023-24.pdf",
    "data/data/CBE Annual Report 2018-19.pdf",
    
    # Class B: Scanned Government/Legal
    "data/data/Audit Report - 2023.pdf", # Scanned
    "data/data/2013-E.C-Audit-finding-information.pdf", # Scanned
    "data/data/DBE ANNUAL REPORT 2022-23.pdf", # Partially Scanned
    
    # Class C: Technical Assessment / Mixed Layout
    "data/data/fta_performance_survey_final_report_2022.pdf",
    "data/data/20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf",
    "data/data/Company_Profile_2024_25.pdf",
    
    # Class D: Structured / Numerical / Tax 
    "data/data/tax_expenditure_ethiopia_2021_22.pdf",
    "data/data/Consumer Price Index August 2025.pdf",
    "data/data/Consumer Price Index September 2025.pdf"
]

print("=== STARTING FULL PIPELINE BATCH ===")

# 1. Triage all
print("\\n--- Phase 1: Triage ---")
for doc in docs:
    if os.path.exists(doc):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        subprocess.run(["python3", "src/agents/triage.py", doc], env=env)
    else:
        print(f"Missing doc: {doc}")

# 2. Extract all
# run_extraction.py reads all unextracted profiles and extracts them
print("\n--- Phase 2: Extraction ---")
subprocess.run(["python3", "run_extraction.py"])

# 3. Chunk all completed extractions
# 4. Index and Ingest all 
print("\n--- Phase 3 & 4: Semantic Chunking, Indexing, and QA Generation ---")
extracted_files = glob.glob(".refinery/extractions/*.json")
# Sort files to ensure consistent processing order, useful for debugging
extracted_files.sort() 

for i, extraction_file in enumerate(extracted_files):
    if extraction_file.endswith("_ldu.json"): # Skip LDUs
        continue

    # Derive original document path for logging
    # Assuming extraction_file is like .refinery/extractions/data_data_docname.pdf.json
    # and original doc is data/data/docname.pdf
    base_name = os.path.basename(extraction_file).replace(".json", "")
    doc_path_parts = base_name.split('_')
    # Reconstruct original path, assuming 'data_data' prefix
    if len(doc_path_parts) > 2 and doc_path_parts[0] == 'data' and doc_path_parts[1] == 'data':
        doc_path = os.path.join(doc_path_parts[0], doc_path_parts[1], '_'.join(doc_path_parts[2:]))
    else:
        doc_path = base_name # Fallback if naming convention is different

    # Phase 3: Semantic Chunking & Indexing
    print(f"[{i+1}/{len(extracted_files)}] Running Phase 3 (Chunking & Index) for {doc_path}...")
    try:
        subprocess.run(["python3", "run_indexer.py", "--json_path", extraction_file], check=True)
        print(f"[{i+1}/{len(extracted_files)}] Phase 3 Success: {doc_path}")
    except subprocess.CalledProcessError as e:
        print(f"[{i+1}/{len(extracted_files)}] Phase 3 Failed for {doc_path}: {e}")
        continue
        
    # Phase 4: Vector Store & FactTable Synthesis (Generate QA pairs)
    print(f"[{i+1}/{len(extracted_files)}] Running Phase 4 (Knowledge Graph) for {doc_path}...")
    try:
        subprocess.run(["python3", "run_query_agent.py", "--json_path", extraction_file, "--batch"], check=True)
        print(f"[{i+1}/{len(extracted_files)}] Phase 4 Success: QA Batch generated for {doc_path}")
    except subprocess.CalledProcessError as e:
        print(f"[{i+1}/{len(extracted_files)}] Phase 4 Failed for {doc_path}: {e}")
        continue

print("\n=== BATCH PIPELINE COMPLETE ===")
