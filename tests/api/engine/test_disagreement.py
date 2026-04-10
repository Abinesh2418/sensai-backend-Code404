from api.engine.disagreement import detect_disagreements

def test_agreement_within_15_percent():
    result = detect_disagreements({"keyword": 0.80, "semantic": 0.85, "llm": 0.82})
    assert len(result) == 0

def test_minor_disagreement_15_to_30():
    result = detect_disagreements({"keyword": 0.50, "llm": 0.75})
    assert len(result) == 1
    assert result[0]["severity"] == "minor"
    assert result[0]["resolution"] == "proceed_weighted_average"

def test_major_disagreement_above_30():
    result = detect_disagreements({"keyword": 0.30, "llm": 0.90})
    assert len(result) == 1
    assert result[0]["severity"] == "major"
    assert result[0]["resolution"] == "queue_human_review"

def test_multiple_pairs():
    result = detect_disagreements({"keyword": 0.30, "semantic": 0.35, "llm": 0.90})
    assert len(result) == 2
    assert all(d["severity"] == "major" for d in result)

def test_none_scores_are_skipped():
    result = detect_disagreements({"keyword": 0.80, "semantic": None, "llm": 0.85})
    assert len(result) == 0
