import subprocess
import os
import json
import time

new_docs = [
    "data/data/2021_Audited_Financial_Statement_Report.pdf",
    "data/data/CBE Annual Report 2018-19.pdf",
    "data/data/20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf",
    "data/data/Company_Profile_2024_25.pdf",
    "data/data/Security_Vulnerability_Disclosure_Standard_Procedure_1.pdf",
    "data/data/Consumer Price Index August 2025.pdf",
    "data/data/Consumer Price Index September 2025.pdf",
    "data/data/2013-E.C-Audit-finding-information.pdf"
]

print("Starting targeted triage for 8 new documents...")
for doc in new_docs:
    print(f"\nTriaging {doc}...")
    subprocess.run(["python3", "src/agents/triage.py", doc])

# Wait a second for filesystem flush
time.sleep(2)

print("\nFinding document IDs from the newly created profiles...")
doc_ids = []
basenames = [os.path.basename(doc) for doc in new_docs]
for p in os.listdir(".refinery/profiles"):
    if p.endswith(".json"):
        with open(os.path.join(".refinery/profiles", p)) as f:
            data = json.load(f)
            if data['file'] in basenames:
                doc_ids.append(data['document_id'])

print(f"Found {len(doc_ids)} matching profiles. Starting targeted extraction...")
for doc_id in doc_ids:
    print(f"\nExtracting document ID: {doc_id}...")
    subprocess.run(["python3", "run_single.py", doc_id])

print("\nBatch triage and extraction complete.")
