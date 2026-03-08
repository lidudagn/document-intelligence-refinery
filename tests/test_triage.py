import pytest
from src.agents.triage import TriageAgent
from src.models import PageMetrics
from src.models.profile import OriginType, LayoutComplexity, ExtractionCost

@pytest.fixture
def triage_agent():
    return TriageAgent()

def test_origin_scanned_classification(triage_agent):
    """Assert pure SCANNED_IMAGE logic via data matrix."""
    metrics = [
        PageMetrics(page_number=1, char_density=0.0001, image_area_ratio=0.8, table_count=0, column_count=1, whitespace_ratio=0.1, has_text_layer=False, text_sample=None),
        PageMetrics(page_number=2, char_density=0.0001, image_area_ratio=0.9, table_count=0, column_count=1, whitespace_ratio=0.1, has_text_layer=False, text_sample=None),
    ]
    origin, conf = triage_agent._classify_origin(metrics, is_acroform=False)
    assert origin == OriginType.SCANNED_IMAGE
    assert conf == 1.0

def test_origin_native_classification(triage_agent):
    """Assert NATIVE_DIGITAL baseline assignment."""
    metrics = [
        PageMetrics(page_number=1, char_density=0.02, image_area_ratio=0.1, table_count=1, column_count=1, whitespace_ratio=0.2, has_text_layer=True, text_sample="Test native content."),
    ]
    origin, conf = triage_agent._classify_origin(metrics, is_acroform=False)
    assert origin == OriginType.NATIVE_DIGITAL
    assert conf == 1.0

def test_origin_mixed_edge_case(triage_agent):
    """Assert MIXED is detected for hybrid scanned/native documents."""
    metrics = [
        PageMetrics(page_number=1, char_density=0.0001, image_area_ratio=0.9, table_count=0, column_count=1, whitespace_ratio=0.1, has_text_layer=False, text_sample=None),
        PageMetrics(page_number=2, char_density=0.02, image_area_ratio=0.1, table_count=1, column_count=1, whitespace_ratio=0.2, has_text_layer=True, text_sample="Native text."),
    ]
    # S = 0.5, N = 0.5 -> 0.30 <= S < 0.70 is MIXED
    origin, conf = triage_agent._classify_origin(metrics, is_acroform=False)
    assert origin == OriginType.MIXED
    assert conf == 0.5

def test_origin_form_fillable(triage_agent):
    """Assert AcroForm flag aggressively overrides standard matrices."""
    # Give it purely scanned metrics, but set acroform to True
    metrics = [
        PageMetrics(page_number=1, char_density=0.0001, image_area_ratio=0.9, table_count=0, column_count=1, whitespace_ratio=0.1, has_text_layer=False, text_sample=None),
    ]
    origin, conf = triage_agent._classify_origin(metrics, is_acroform=True)
    assert origin == OriginType.FORM_FILLABLE
    assert conf == 1.0

def test_layout_table_heavy(triage_agent):
    """Ensure TABLE_HEAVY triggers overriding default SINGLE_COLUMN."""
    metrics = [
        PageMetrics(page_number=1, char_density=0.01, image_area_ratio=0.0, table_count=2, column_count=1, whitespace_ratio=0.5, has_text_layer=True),
        PageMetrics(page_number=2, char_density=0.01, image_area_ratio=0.0, table_count=1, column_count=1, whitespace_ratio=0.5, has_text_layer=True),
    ]
    layout, conf = triage_agent._classify_layout(metrics)
    assert layout == LayoutComplexity.TABLE_HEAVY
    assert conf == 1.0

def test_language_fallback(triage_agent):
    """Assert unknown is safely returned on SCANNED_IMAGE origins."""
    metrics = [PageMetrics(page_number=1, char_density=0.0001, image_area_ratio=0.9, table_count=0, column_count=1, whitespace_ratio=0.1, has_text_layer=False, text_sample=None)]
    lang, conf = triage_agent._detect_language(metrics, OriginType.SCANNED_IMAGE)
    assert lang == "unknown"
    assert conf == 0.0

def test_domain_hint_financial(triage_agent):
    """Assert weighted domain scoring triggers FINANCIAL via keywords."""
    metrics = [
        PageMetrics(page_number=1, char_density=0.02, image_area_ratio=0.0, table_count=0, column_count=1, whitespace_ratio=0.5, has_text_layer=True, text_sample="The income statement highlights our revenue and fiscal health without jurisdiction issues."),
    ]
    # Hits: "income statement" (5), "revenue" (5), "fiscal" (5), "jurisdiction" (5 for legal, but financial is 15 -> wins)
    domain, conf = triage_agent._detect_domain(metrics)
    assert domain == "financial"
    assert round(conf, 2) > 0.0 # Just assert we got a valid confidence measurement

def test_cost_routing_escalation(triage_agent):
    """Assert intelligent routing escalation based on Origin certitude."""
    # A single column native doc, but poor confidence -> escalate to LAYOUT_MODEL
    cost = triage_agent._route_extraction(OriginType.NATIVE_DIGITAL, o_conf=0.4, layout=LayoutComplexity.SINGLE_COLUMN, l_conf=1.0)
    assert cost == ExtractionCost.NEEDS_LAYOUT_MODEL

def test_layout_single_page_edge(triage_agent):
    """Assert empty document arrays safely return Native/Single Column defaults."""
    metrics = []
    origin, o_conf = triage_agent._classify_origin(metrics, is_acroform=False)
    layout, l_conf = triage_agent._classify_layout(metrics)
    
    assert origin == OriginType.NATIVE_DIGITAL
    assert o_conf == 0.0
    assert layout == LayoutComplexity.SINGLE_COLUMN
    assert l_conf == 0.0

def test_layout_figure_heavy(triage_agent):
    """Assert FIGURE_HEAVY overrides SINGLE_COLUMN when images dominate dense text pages."""
    metrics = [
        PageMetrics(page_number=1, char_density=0.02, image_area_ratio=0.8, table_count=0, column_count=1, whitespace_ratio=0.1, has_text_layer=True, text_sample="Chart caption"),
    ]
    layout, conf = triage_agent._classify_layout(metrics)
    assert layout == LayoutComplexity.FIGURE_HEAVY
    assert conf == 1.0

def test_layout_multi_column(triage_agent):
    """Assert MULTI_COLUMN classification on clustered X coordinates."""
    metrics = [
        PageMetrics(page_number=1, char_density=0.03, image_area_ratio=0.0, table_count=0, column_count=3, whitespace_ratio=0.5, has_text_layer=True, text_sample="Column 1\tColumn 2\tColumn 3"),
    ]
    layout, conf = triage_agent._classify_layout(metrics)
    assert layout == LayoutComplexity.MULTI_COLUMN
    assert conf == 1.0

def test_domain_activation_fallback(triage_agent):
    """Assert domain falls back to GENERAL if keywords don't hit the minimum 10 threshold."""
    metrics = [
        # Only one hit for 'revenue' (weight 5)
        PageMetrics(page_number=1, char_density=0.02, image_area_ratio=0.0, table_count=0, column_count=1, whitespace_ratio=0.5, has_text_layer=True, text_sample="Just a small revenue mention."),
    ]
    domain, conf = triage_agent._detect_domain(metrics)
    assert domain == "general"
    assert conf == 0.0
