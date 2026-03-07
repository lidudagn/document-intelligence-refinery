# Week 3: Document Intelligence Refinery

An enterprise-scale PDF extraction pipeline designed to reliably convert uncooperative business documents into queryable structured data.

## Architecture Highlights
This system acts as a deterministic funnel, handling everything from clean digital PDFs to degraded scanned images utilizing a tiered cost-extraction strategy.

1. **Triage Agent:** Analyzes visual layout and text density to classify documents as `native_digital`, `scanned_image`, or `mixed_layout`.
2. **Extraction Router:** Executes extraction strategies efficiently using Confidence-Aware Routing:
   - **Strategy A (Fast Text):** `pdfplumber`. Target cost < $0.001 per page.
   - **Strategy B (Layout-Aware):** `Docling`. Runs in an isolated subprocess to prevent OOM errors.
   - **Strategy C (Vision Augmented):** Multimodal LLM (Gemini/OpenAI) for complex graphics or degraded scans, constrained by a strict `BudgetGuard`.
3. **Semantic Chunking:** Converts extracted blocks into Logical Document Units (LDUs) using content-hash deduplication and constitution rules that forbid splitting tables or separating figures from captions.
4. **Agentic Indexing:** Hierarchically structures LDUs with LLM-generated summaries (or heuristic fallbacks) while syncing chunks to a local ChromaDB instance.
5. **Auditable Query Interface:** An LLM-driven query agent equipped with a `FactTable` for deterministic SQL retrieval and `VectorSearch` for semantic retrieval, outputting responses tied strictly to an irrefutable `ProvenanceChain` (Document -> Page -> BBox).

## Setup & Installation

**Prerequisites:** Python 3.10+

```bash
# Clone and enter the repository
git clone <your-repo-url>
cd document-intelligence-refinery

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root:

```ini
GOOGLE_API_KEY="your-gemini-key"
OPENAI_API_KEY="your-openai-key" # Optional, for Vision/Chunk indexing
OPENROUTER_API_KEY="your-openrouter-key" # Recommended for avoiding rate limits
INDEXER_LLM_MODEL="openrouter/qwen/qwen-2.5-72b-instruct"
MAX_VLM_COST_PER_DOCUMENT="0.10"
```

## Running the Pipeline

You can run individual phases for debugging or the full batch pipeline to process a corpus.

### Running Single Documents
```bash
# Phase 1: Triage
python run_triage.py path/to/document.pdf

# Phase 2: Extraction
python run_extraction.py <document_sha256_id>

# Phase 3 & 4: Chunking and Indexing
python run_indexer.py --json_path .refinery/extractions/<document_id>.json
python run_query_agent.py --json_path .refinery/pageindex/<document_id>_index.json
```

### Running Batch (Full Corpus)
Execute the provided script to process the entire `/data/data/` corpus end-to-end:
```bash
python run_full_batch.py
```

### Running the Query Interface
To chat with the processed documents:
```bash
python run_query_agent.py --json_path <any_processed_json_path> --interactive
```

## Testing
The pipeline is fully unit-tested to guarantee structural extraction bounds:
```bash
pytest tests/
```
