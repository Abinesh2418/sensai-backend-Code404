from api.engine.aggregator import aggregate_text_signals

def test_text_aggregation_without_human():
    result = aggregate_text_signals(
        keyword_normalized=0.80, semantic_normalized=0.70,
        llm_normalized=0.90, human_normalized=None, max_score=30.0,
    )
    expected_score = 0.835 * 30.0  # 0.80*0.15 + 0.70*0.25 + 0.90*0.60 = 0.835
    assert abs(result["final_score"] - expected_score) < 0.01
    assert result["weights_used"]["keyword"] == 0.15
    assert result["weights_used"]["semantic"] == 0.25
    assert result["weights_used"]["llm"] == 0.60

def test_text_aggregation_with_human():
    result = aggregate_text_signals(
        keyword_normalized=0.80, semantic_normalized=0.70,
        llm_normalized=0.90, human_normalized=0.60, max_score=30.0,
    )
    expected_score = 0.755 * 30.0  # 0.80*0.10+0.70*0.15+0.90*0.40+0.60*0.35
    assert abs(result["final_score"] - expected_score) < 0.01
    assert result["weights_used"]["human"] == 0.35

def test_missing_evaluator_redistributes_weight():
    result = aggregate_text_signals(
        keyword_normalized=0.80, semantic_normalized=None,
        llm_normalized=0.90, human_normalized=None, max_score=10.0,
    )
    # keyword(0.15)+llm(0.60)=0.75 → keyword=0.20, llm=0.80
    expected = (0.80*0.20 + 0.90*0.80) * 10.0  # 0.88 * 10 = 8.8
    assert abs(result["final_score"] - expected) < 0.01

def test_llm_skipped_tier1_only():
    result = aggregate_text_signals(
        keyword_normalized=0.90, semantic_normalized=0.85,
        llm_normalized=None, human_normalized=None, max_score=10.0,
    )
    # keyword(0.15)+semantic(0.25)=0.40 → keyword=0.375, semantic=0.625
    expected = (0.90*0.375 + 0.85*0.625) * 10.0
    assert abs(result["final_score"] - expected) < 0.01

def test_explanation_contains_signal_details():
    result = aggregate_text_signals(
        keyword_normalized=0.80, semantic_normalized=0.70,
        llm_normalized=0.90, human_normalized=None, max_score=30.0,
    )
    assert "signals" in result
    assert len(result["signals"]) == 3
    assert result["method"] == "spec_weighted_average"
