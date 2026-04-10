from api.engine.confidence_gate import should_run_llm, compute_tier1_confidence

def test_high_confidence_low_stakes_skips_llm():
    assert should_run_llm(keyword_normalized=0.90, semantic_normalized=0.92, stakes="low") is False

def test_high_confidence_high_stakes_runs_llm():
    assert should_run_llm(keyword_normalized=0.95, semantic_normalized=0.95, stakes="high") is True

def test_medium_confidence_runs_llm():
    assert should_run_llm(keyword_normalized=0.60, semantic_normalized=0.70, stakes="low") is True

def test_low_confidence_runs_llm():
    assert should_run_llm(keyword_normalized=0.20, semantic_normalized=0.25, stakes="low") is True

def test_confidence_formula():
    conf = compute_tier1_confidence(keyword_normalized=0.80, semantic_normalized=0.80)
    # agreement = 1 - |0.80 - 0.80| = 1.0
    # extremity = (0.80 + 0.80) / 2 = 0.80
    # confidence = 1.0 * 0.6 + 0.80 * 0.4 = 0.92
    assert abs(conf - 0.92) < 0.01

def test_confidence_with_disagreement():
    conf = compute_tier1_confidence(keyword_normalized=0.90, semantic_normalized=0.50)
    # agreement = 1 - 0.40 = 0.60, extremity = 0.70
    # confidence = 0.60 * 0.6 + 0.70 * 0.4 = 0.64
    assert abs(conf - 0.64) < 0.01

def test_missing_semantic_always_runs_llm():
    assert should_run_llm(keyword_normalized=0.95, semantic_normalized=None, stakes="low") is True
