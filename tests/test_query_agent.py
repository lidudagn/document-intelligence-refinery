from unittest.mock import MagicMock
from src.agents.query_agent import QueryAgent


def test_provenance_chain_assembly():
    vstore = MagicMock()
    fact_db = MagicMock()
    config = {"query_agent": {"min_retrieval_confidence": 0.0}}
    agent = QueryAgent(vstore, fact_db, config)
    
    chunks = [
        {"content": "Text 1", "metadata": {"page_refs": "1", "content_hash": "h1"}, "score": 0.9},
        {"content": "Text 2", "metadata": {"page_refs": "2", "content_hash": "h2"}, "score": 0.8}
    ]
    
    # Mock LLM response
    agent.cache.get = MagicMock(return_value="Mocked Answer")
    
    chain = agent._synthesize_with_llm("query", chunks, [], "DocName")
    
    assert chain.answer == "Mocked Answer"
    assert len(chain.citations) == 2
    assert chain.citations[0].page_number == 1
    assert chain.citations[1].content_hash == "h2"
    assert chain.confidence_level == 0.85  # average of 0.9 and 0.8


def test_retrieval_confidence_gate():
    vstore = MagicMock()
    fact_db = MagicMock()
    config = {"query_agent": {"min_retrieval_confidence": 0.6}}
    agent = QueryAgent(vstore, fact_db, config)
    
    # Setup mock hybrid search to return low-scoring chunks
    agent._hybrid_search = MagicMock(return_value=[
        {"content": "Irrelevant", "score": 0.2, "metadata": {"content_hash": "h1"}}
    ])
    agent._structured_query = MagicMock(return_value=[])
    
    chain = agent.query("test query", "doc1", MagicMock())
    
    assert chain.is_verifiable is False
    assert "sufficient confidence" in chain.answer
    assert chain.confidence_level == 0.2


def test_sql_direct_shortcut():
    vstore = MagicMock()
    fact_db = MagicMock()
    config = {"query_agent": {"sql_direct_answer": True}}
    agent = QueryAgent(vstore, fact_db, config)
    
    agent._structured_query = MagicMock(return_value=[
        {
            "metric": "Revenue", "value": "$4.2B", "period": "FY2023", 
            "page_number": 1, "content_hash": "h1", "bbox": "[1,2,3,4]"
        }
    ])
    
    chain = agent.query("What is the revenue?", "doc1", MagicMock())
    
    assert chain.is_verifiable is True
    assert "Revenue: $4.2B" in chain.answer
    assert chain.retrieval_method == "structured_query_direct"
