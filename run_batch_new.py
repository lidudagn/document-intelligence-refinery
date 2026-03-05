import subprocess
import os
import json

new_docs = [
    "2021_Audited_Financial_Statement_Report.pdf",
    "CBE Annual Report 2018-19.pdf",
    "20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf",
    "Company_Profile_2024_25.pdf",
    "Security_Vulnerability_Disclosure_Standard_Procedure_1.pdf",
    "Consumer Price Index August 2025.pdf",
    "Consumer Price Index September 2025.pdf",
    "2013-E.C-Audit-finding-information.pdf"
]

# Find the document IDs from the profiles
doc_ids = []
for p in os.listdir(".refinery/profiles"):
    if p.endswith(".json"):
        with open(os.path.join(".refinery/profiles", p)) as f:
            data = json.load(f)
            if data['file'] in new_docs:
                doc_ids.append(data['document_id'])

print(f"Found {len(doc_ids)} matching profiles. Starting targeted extraction...")
for doc_id in doc_ids:
    print(f"\nExtracting document ID: {doc_id}...")
    subprocess.run(["python3", "run_single.py", doc_id])

print("\nBatch extraction complete.")
