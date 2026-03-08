from unittest.mock import MagicMock
from src.agents.fact_table import FactTableDB, parse_numeric


def test_heuristic_extraction_from_table_ldu():
    # We test the parser directly
    text1 = "$4.2B"
    assert parse_numeric(text1) == 4200000000.0
    
    text2 = "12.5%"
    assert parse_numeric(text2) == 12.5
    
    text3 = "(1,234.56)"
    assert parse_numeric(text3) == -1234.56
    
    text4 = "ETB 500M"
    assert parse_numeric(text4) == 500000000.0


def test_sqlite_insert_and_query(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = FactTableDB(db_path)
    
    # Needs FactRecord import but we mock the inputs for isolation
    from src.models.fact_record import FactRecord
    
    fact = FactRecord(
        fact_id="123",
        document_id="doc1",
        metric="Revenue",
        entity="CBE",
        period="FY2023",
        value="$4.2B",
        unit="USD",
        numeric_value=4200000000.0,
        page_number=1,
        content_hash="hash123",
        confidence=0.9
    )
    
    db.insert_facts([fact])
    
    # Query
    results = db.search_facts("Revenue")
    assert len(results) == 1
    assert results[0]["metric"] == "Revenue"
    assert results[0]["numeric_value"] == 4200000000.0
    
    db.close()

def test_sql_injection_guard(tmp_path):
    import pytest
    db_path = str(tmp_path / "test.db")
    db = FactTableDB(db_path)
    
    with pytest.raises(ValueError, match="Only SELECT queries are allowed"):
        db.query_facts("DROP TABLE facts")
        
    with pytest.raises(ValueError, match="Dangerous SQL keyword detected"):
        db.query_facts("SELECT * FROM facts WHERE 1=1; DROP TABLE facts")
        
    db.close()
