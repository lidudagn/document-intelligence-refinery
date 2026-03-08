from unittest.mock import MagicMock
from src.agents.auditor import AuditAgent
from src.models.audit_result import AuditVerdict


def test_heuristic_verification():
    agent = AuditAgent(MagicMock(), MagicMock(), {})
    
    sql_hits = [
        {"metric": "Revenue", "value": "$4.2B", "numeric_value": 4200000000.0, "page_number": 1}
    ]
    
    # Exact match
    res1 = agent._heuristic_verify("Revenue was $4.2B", sql_hits, [])
    assert res1["verdict"] == AuditVerdict.VERIFIED.value
    
    # Close match (e.g. rounded claim vs exact fact)
    res2 = agent._heuristic_verify("Revenue was $4.15B", sql_hits, [])
    assert res2["verdict"] == AuditVerdict.PARTIALLY_VERIFIED.value
    
    # Contradicted / Unverifiable - no numeric match but semantically retrieved
    res3 = agent._heuristic_verify("Revenue was $10B", sql_hits, [{"content": "Rev"}])
    assert res3["verdict"] == AuditVerdict.PARTIALLY_VERIFIED.value # because vector found something, leaving to human
    
    # No evidence anywhere
    res4 = agent._heuristic_verify("Revenue was $10B", [], [])
    assert res4["verdict"] == AuditVerdict.UNVERIFIABLE.value


def test_auditor_confidence_scoring(monkeypatch):
    import src.agents.auditor as auditor_module
    monkeypatch.setattr(auditor_module, "LITELLM_AVAILABLE", True)
    
    agent = AuditAgent(MagicMock(), MagicMock(), {})
    
    # Mock retrieval
    agent._vector_evidence = MagicMock(return_value=[{"content": "test", "metadata": {}}])
    agent._sql_evidence = MagicMock(return_value=[{"metric": "test"}])
    agent._llm_verify = MagicMock(return_value={"verdict": AuditVerdict.VERIFIED.value, "reasoning": "ok"})
    
    res = agent.verify("test claim", "doc1")
    
    assert res.verdict == AuditVerdict.VERIFIED
    # 2 sources returned evidence -> confidence should be 0.7 based on logic map
    assert res.confidence == 0.7
