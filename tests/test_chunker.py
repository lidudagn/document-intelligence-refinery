import pytest
from src.agents.chunker import ChunkingEngine, ChunkValidator
from src.models.extracted_document import ExtractedDocument, ExtractedPage, TextBlock, TableBlock, BoundingBox
from src.models.ldu import ChunkType

@pytest.fixture
def chunker():
    config = {
        "chunking": {
            "max_tokens_per_chunk": 50,
            "enforce_rules": True
        }
    }
    return ChunkingEngine(config)

def test_chunking_deterministic_hash(chunker):
    doc = ExtractedDocument(
        document_id="doc_1",
        source_path="x.pdf",
        pages=[
            ExtractedPage(
                page_number=0,  # 0-indexed (real extraction uses 0-indexed)
                strategy_used="fast_text",
                blocks=[
                    TextBlock(page_number=0, text="Hello world", content_hash="x", bbox=BoundingBox(x0=0, top=0, x1=10, bottom=10))
                ]
            )
        ]
    )
    ldus = chunker.process_document(doc)
    assert len(ldus) == 1
    # Chunker converts 0-indexed -> 1-indexed, so page_refs=[1]
    assert ldus[0].page_refs == [1]
    assert ldus[0].content_hash == chunker._compute_hash([1], BoundingBox(x0=0, top=0, x1=10, bottom=10), "Hello world")

def test_tables_must_not_split(chunker):
    """Rule 1: Tables stay intact even if token count > limits theoretically."""
    doc = ExtractedDocument(
        document_id="doc_2",
        source_path="x.pdf",
        pages=[
            ExtractedPage(
                page_number=0,
                strategy_used="fast_text",
                blocks=[
                    TableBlock(
                        page_number=0,
                        headers=["A", "B"],
                        rows=[["1", "2"] * 20], # make it large
                        content_hash="y"
                    )
                ]
            )
        ]
    )
    ldus = chunker.process_document(doc)
    assert len(ldus) == 1
    assert ldus[0].chunk_type == ChunkType.TABLE
    assert ldus[0].page_refs == [1]  # 0-indexed -> 1-indexed
    
def test_headers_propagate(chunker):
    """Rule 4: Section headers become parent_section metadata."""
    doc = ExtractedDocument(
        document_id="doc_3",
        source_path="x.pdf",
        pages=[
            ExtractedPage(
                page_number=0,
                strategy_used="fast_text",
                blocks=[
                    TextBlock(page_number=0, text="INTRODUCTION", content_hash="h1"),
                    TextBlock(page_number=0, text="This is the first paragraph.", content_hash="p1")
                ]
            )
        ]
    )
    ldus = chunker.process_document(doc)
    assert len(ldus) == 2
    assert ldus[1].parent_section == "INTRODUCTION"
    assert all(ldu.page_refs == [1] for ldu in ldus)
