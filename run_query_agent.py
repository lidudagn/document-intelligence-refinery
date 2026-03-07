"""
Phase 4 End-to-End Runner
Builds the knowledge graph + FactTable from an extraction, then runs
the Query Agent or Audit Agent.

Modes:
  --interactive : REPL for live queries
  --batch       : Generate batch Q&A examples
  --audit       : Verify a specific claim
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load .env before other imports that might depend on env vars
load_dotenv()

# Sync GOOGLE_API_KEY to GEMINI_API_KEY for LiteLLM compatibility
if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

from src.models.extracted_document import ExtractedDocument
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.vector_store import VectorStoreClient
from src.agents.fact_table import FactTableExtractor, FactTableDB
from src.agents.query_agent import QueryAgent
from src.agents.auditor import AuditAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_extraction(json_path: str) -> Optional[ExtractedDocument]:
    path = Path(json_path)
    if not path.exists():
        logger.error(f"Extraction file not found: {json_path}")
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return ExtractedDocument(**data)
    except Exception as e:
        logger.error(f"Failed to load ExtractedDocument: {e}")
        return None


def run_pipeline(json_path: str, config: dict = None) -> tuple:
    """Run Chunking -> Indexing -> Vector Store -> FactTable."""
    doc = load_extraction(json_path)
    if not doc:
        sys.exit(1)

    print(f"\n--- Processing {doc.document_id} ---")

    # 1. Chunking
    chunker = ChunkingEngine(config)
    ldus = chunker.process_document(doc)
    print(f"[1] Chunking: {len(ldus)} chunks generated")

    # 2. Indexing
    indexer = PageIndexBuilder(config)
    page_index = indexer.build_index(doc.document_id, ldus)
    print(f"[2] Indexing: PageIndex tree built with {len(page_index.root_sections)} root sections")

    # 3. Vector Store
    vstore = VectorStoreClient(config)
    vstore.ingest_ldus(doc.document_id, ldus)
    print(f"[3] Vector Store: LDUs ingested via dedup")

    # 4. FactTable
    fact_db_path = f".refinery/fact_tables/{doc.document_id}.db"
    extractor = FactTableExtractor(config)
    facts = extractor.extract_from_ldus(doc.document_id, ldus, fact_db_path)
    print(f"[4] FactTable: Extracted {len(facts)} structured facts")
    
    fact_db = FactTableDB(fact_db_path)
    
    doc_name = getattr(doc, "source_filename", doc.document_id)
    return doc.document_id, doc_name, ldus, page_index, vstore, fact_db


def interactive_mode(
    doc_id: str, doc_name: str, page_index, vstore: VectorStoreClient, fact_db: FactTableDB, config: dict
):
    """Interactive REPL terminal."""
    agent = QueryAgent(vstore, fact_db, config)
    print("\n" + "="*50)
    print(f"Refinery Query Agent REPL — Document: {doc_name}")
    print("Type 'exit' or 'quit' to exit.")
    print("="*50 + "\n")

    while True:
        try:
            query = input(f"\n[{doc_name}] Query > ")
            query = query.strip()
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue

            print("Thinking...\n")
            chain = agent.query(query, doc_id, page_index, doc_name)

            print(f"ANSWER:\n{chain.answer}\n")
            
            if chain.citations:
                print("CITATIONS:")
                for i, cit in enumerate(chain.citations):
                    print(f" [{i+1}] Page {cit.page_number} | Match: {cit.extracted_text[:60].strip()}...")
                
            print(f"\n[Meta: Confidence={chain.confidence_level:.2f} | Method={chain.retrieval_method}]")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


def batch_mode(
    doc_id: str, doc_name: str, page_index, vstore: VectorStoreClient, fact_db: FactTableDB, config: dict
):
    """Generate 3 Q&A examples per document and save as artifact."""
    agent = QueryAgent(vstore, fact_db, config)
    
    # Generic questions that apply to most business/government docs
    queries = [
        "What is the overall summary or primary objective of this document?",
        "What are the key financial or numerical highlights mentioned?",
        "What are the main risks, challenges, or recommendations identified?",
    ]
    
    artifact_path = Path(f".refinery/qa_examples/{doc_id}_qa.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "document_id": doc_id,
        "document_name": doc_name,
        "qa_pairs": []
    }
    
    print(f"\nRunning batch Q&A generation ({len(queries)} queries)...")
    for q in queries:
        print(f"  Query: {q}")
        chain = agent.query(q, doc_id, page_index, doc_name)
        results["qa_pairs"].append(chain.model_dump())
        
    with open(artifact_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nBatch complete. Artifact saved to: {artifact_path}")


def audit_mode(
    claim: str, doc_id: str, doc_name: str, vstore: VectorStoreClient, fact_db: FactTableDB, config: dict
):
    """Run Auditor pass on a specific claim."""
    auditor = AuditAgent(vstore, fact_db, config)
    print(f"\nAuditing Claim: \"{claim}\"")
    print("Verifying against vector store and SQL fact table...\n")
    
    result = auditor.verify(claim, doc_id, doc_name)
    
    print(f"VERDICT: {result.verdict.value.upper()}")
    print(f"Confidence: {result.confidence:.2f}")
    if result.reasoning:
        print(f"Reasoning: {result.reasoning}")
        
    if result.supporting_citations:
        print("\nSupporting Evidence:")
        for idx, cit in enumerate(result.supporting_citations):
            print(f"  [{idx+1}] Page {cit.page_number} Hash: {cit.content_hash[:8]}")
            
    if result.contradicting_evidence:
        print("\nContradicting Evidence:")
        for idx, cit in enumerate(result.contradicting_evidence):
            print(f"  [{idx+1}] Page {cit.page_number} Hash: {cit.content_hash[:8]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Phase 4 Query Pipeline")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the ExtractedDocument JSON")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive REPL")
    parser.add_argument("--batch", action="store_true", help="Generate batch Q&A examples")
    parser.add_argument("--audit", type=str, help="Verify a specific claim against the corpus")
    
    args = parser.parse_args()
    
    if not (args.interactive or args.batch or args.audit):
        print("Error: Must specify one of: --interactive, --batch, --audit")
        sys.exit(1)
        
    import yaml
    try:
        with open("rubric/extraction_rules.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}
        
    doc_id, doc_name, ldus, page_index, vstore, fact_db = run_pipeline(args.json_path, config)
    
    if args.interactive:
        interactive_mode(doc_id, doc_name, page_index, vstore, fact_db, config)
    elif args.batch:
        batch_mode(doc_id, doc_name, page_index, vstore, fact_db, config)
    if args.audit:
        audit_mode(args.audit, doc_id, doc_name, vstore, fact_db, config)
