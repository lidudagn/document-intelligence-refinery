import subprocess
import time

docs = [
    # Financial/Annual Reports (Native)
    "data/data/2021_Audited_Financial_Statement_Report.pdf",
    "data/data/CBE Annual Report 2018-19.pdf",
    
    # Technical Assessments / General (Mixed)
    "data/data/20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf",
    "data/data/Company_Profile_2024_25.pdf",
    
    # Legal / Framework / Policy (Mixed or Native)
    "data/data/Security_Vulnerability_Disclosure_Standard_Procedure_1.pdf",
    
    # Structured / Numerical (Native)
    "data/data/Consumer Price Index August 2025.pdf",
    "data/data/Consumer Price Index September 2025.pdf",
    
    # Scanned / Image-heavy (Scanned)
    "data/data/2013-E.C-Audit-finding-information.pdf"
]

print("Starting batch triage...")
for doc in docs:
    print(f"\nTriaging {doc}...")
    subprocess.run(["python3", "src/agents/triage.py", doc])

print("\nStarting batch extraction...")
subprocess.run(["python3", "run_extraction.py"])

print("\nBatch processing complete.")
