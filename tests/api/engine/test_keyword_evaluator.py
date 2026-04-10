from api.engine.keyword_evaluator import evaluate_keywords

def test_exact_match():
    result = evaluate_keywords("recursion uses a base case and stack", "recursion base case stack", max_score=10.0)
    assert result["coverage"] == 1.0
    assert result["score"] == 10.0
    assert len(result["missing_keywords"]) == 0

def test_partial_match():
    result = evaluate_keywords("recursion is important", "recursion base case stack overflow", max_score=10.0)
    assert 0 < result["coverage"] < 1.0
    assert len(result["missing_keywords"]) > 0

def test_stemming_matches_variants():
    result = evaluate_keywords("the algorithms are running efficiently", "algorithm run efficiency", max_score=10.0)
    assert result["matched_count"] >= 3

def test_empty_reference():
    result = evaluate_keywords("some answer", "", max_score=10.0)
    assert result["coverage"] == 0.0
    assert result["score"] == 0.0

def test_empty_answer():
    result = evaluate_keywords("", "recursion base case", max_score=10.0)
    assert result["coverage"] == 0.0

def test_score_scales_to_max():
    result = evaluate_keywords("recursion base", "recursion base case stack", max_score=30.0)
    assert result["score"] == result["coverage"] * 30.0

def test_confidence_is_always_0_95():
    result = evaluate_keywords("test", "test", max_score=10.0)
    assert result["confidence"] == 0.95

def test_case_insensitive():
    result = evaluate_keywords("RECURSION BASE CASE", "recursion base case", max_score=10.0)
    assert result["coverage"] == 1.0
