# Document Intelligence Refinery

Production-grade, multi-stage agentic pipeline that ingests heterogeneous document corpora and emits structured, queryable, spatially-indexed knowledge.

## 🏗 Architecture Overview

The refinery follows a 5-stage agentic architecture designed for enterprise-scale document extraction:

1.  **Triage Agent**: Document classification (origin detection, layout complexity, language/domain analysis). Produces a `DocumentProfile`.
2.  **Structure Extraction Layer**: A multi-strategy router that selects between:
    *   **Fast Text**: `pdfplumber` for native digital prose.
    *   **Layout-Aware**: `Docling` for complex tables and multi-column layouts.
    *   **Vision-Augmented**: Multimodal VLMs (Gemini/GPT-4o) for scanned documents or low-confidence pages.
3.  **Semantic Chunking Engine**: (Phase 3) Converts structured extraction into Logical Document Units (LDUs) that preserve semantic context (tables, headers, figure-caption pairs).
4.  **PageIndex Builder**: (Phase 4) Builds a hierarchical navigation tree for LLM-based document traversal.
5.  **Query Interface Agent**: (Phase 4) A LangGraph agent with spatial provenance citations (page refs + bounding boxes).

## 🚀 Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Configure Environment
cp .env.example .env
# Edit .env and provide your OPENROUTER_API_KEY
```

## 🛠 Usage

### 1. Document Profiling (Triage)
To profile documents in the corpus:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 src/agents/triage.py data/data/your_document.pdf
```

### 2. Multi-Strategy Extraction
To run the extraction engine on a specific document:
```bash
# Via Document ID (Hash)
python3 run_single.py <doc_id_hash>
```

To run extraction on all profiled documents:
```bash
python3 run_extraction.py
```

## 📊 Project Structure

```
src/
  models/       # Pydantic schemas (DocumentProfile, ExtractedDocument, LDU, PageIndex, etc.)
  agents/       # Stage agents (triage, extractor, chunker, indexer, query)
  strategies/   # Extraction strategy implementations (FastText, Layout-Aware, Vision)
rubric/         # Externalized rules and thresholds (extraction_rules.yaml)
.refinery/      # Persistent data layer
  profiles/     # Stage 1 profiling outputs
  extractions/  # Stage 2 extraction outputs
  ledger/       # Performance and cost audit logs
```

## 📄 Documentation

- [DOMAIN_NOTES.md](DOMAIN_NOTES.md): Detailed analysis of extraction strategies, failure modes, and cost analysis.
- [Architecture Diagram](DOMAIN_NOTES.md#4-pipeline-architecture-diagram): Detailed Mermaid diagram of the 5-stage pipeline.
```
