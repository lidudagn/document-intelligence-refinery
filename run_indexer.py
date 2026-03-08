import os
import json
import logging
import argparse
import yaml
from pathlib import Path

from src.models.extracted_document import ExtractedDocument
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.vector_store import VectorStoreClient
from src.agents.query_agent import QueryAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> dict:
    config_path = "rubric/extraction_rules.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Semantic Chunking & Indexing Runner")
    parser.add_argument("--json_path", type=str, required=True, help="Path to a Phase 2 extraction JSON output")
    parser.add_argument("--query", type=str, help="Optional test query for the RAG pipeline")
    args = parser.parse_args()

    # Load extraction result
    if not os.path.exists(args.json_path):
        logger.error(f"File not found: {args.json_path}")
        return

    logger.info(f"Loading document from {args.json_path}")
    with open(args.json_path, "r") as f:
        data = json.load(f)
        doc = ExtractedDocument(**data)

    config = load_config()

    # 1. Semantic Chunking
    logger.info("Step 1: Running Semantic Chunking Engine")
    chunker = ChunkingEngine(config)
    ldus = chunker.process_document(doc)
    
    # Save LDUs out
    os.makedirs(".refinery/ldus/", exist_ok=True)
    ldu_path = f".refinery/ldus/{doc.document_id}_ldus.json"
    with open(ldu_path, "w") as f:
        json.dump([ldu.model_dump() for ldu in ldus], f, indent=2)
    logger.info(f"Successfully generated {len(ldus)} chunks obeying constitutional rules. Saved to {ldu_path}")

    # 2. PageIndex Building
    logger.info("Step 2: Building Hierarchical PageIndex")
    indexer = PageIndexBuilder(config)
    page_index = indexer.build_index(doc.document_id, ldus)
    
    # Save Index out
    os.makedirs(".refinery/pageindex/", exist_ok=True)
    index_path = f".refinery/pageindex/{doc.document_id}_index.json"
    with open(index_path, "w") as f:
        json.dump(page_index.model_dump(), f, indent=2)
    logger.info(f"Successfully generated PageIndex with {len(page_index.root_sections)} sections. Saved to {index_path}")

    # 3. Vector Store Ingestion
    logger.info("Step 3: Ingesting into Vector Store")
    v_client = VectorStoreClient()
    v_client.ingest_ldus(doc.document_id, ldus)

    # 4. Optional querying
    if args.query:
        logger.info(f"Step 4: Executing Query -> '{args.query}'")
        agent = QueryAgent(vector_store=v_client, config=config)
        results = agent.query(args.query, doc.document_id, page_index)
        
        print("\n" + "="*50)
        print(f"Top Results for: '{args.query}'")
        print("="*50)
        for i, cit in enumerate(results.citations):
            print(f"\n[{i+1}] Source Document: {cit.document_name} | Page: {cit.page_number}")
            print(f"Content: {cit.extracted_text[:200]}...")
            
        print(f"\nLLM Synthesized Answer:\n{results.answer}")
            
    logger.info("Phase 3 Execution Complete.")

if __name__ == "__main__":
    main()
