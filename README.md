# Document Intelligence Refinery

Production-grade, multi-stage agentic pipeline that ingests heterogeneous document corpora and emits structured, queryable, spatially-indexed knowledge.

## Pipeline Stages

1. **Triage Agent** — Document classification and profiling
2. **Structure Extraction Layer** — Multi-strategy extraction (Fast Text / Layout-Aware / Vision-Augmented)
3. **Semantic Chunking Engine** — RAG-optimized Logical Document Units
4. **PageIndex Builder** — Hierarchical navigation tree
5. **Query Interface Agent** — Natural language queries with provenance

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env
# Add your API keys to .env
```

## Project Structure

```
src/
  models/       # Pydantic schemas (DocumentProfile, ExtractedDocument, LDU, etc.)
  agents/       # Pipeline agents (triage, extractor, chunker, indexer, query)
  strategies/   # Extraction strategies (fast_text, layout, vision)
rubric/         # Externalized rules and thresholds
scripts/        # Analysis and utility scripts
tests/          # Unit and integration tests
.refinery/      # Runtime outputs (profiles, ledger, pageindex, vectorstore)
```
