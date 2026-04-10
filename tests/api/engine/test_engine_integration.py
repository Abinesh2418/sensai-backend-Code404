from api.engine import run_text_evaluation

def test_full_text_pipeline_all_tiers():
    result = run_text_evaluation(
        user_answer="HyperVerge uses deep learning and computer vision for identity verification",
        reference_answer="HyperVerge is an AI company using deep learning and computer vision for identity verification and KYC",
        llm_score=28.0, llm_max_score=30.0,
        llm_criteria_scores={"Understanding": {"score": 9, "max_score": 10}},
        llm_feedback="Good answer", llm_model="gpt-4.1",
        semantic_normalized=0.85, max_score=30.0, pass_score=18.0, stakes="medium",
    )
    assert result["submission_type"] == "text"
    assert result["final_score"] > 0
    assert result["max_score"] == 30.0
    assert "tier_1" in result
    assert "keyword_matching" in result["tier_1"]
    assert "semantic_similarity" in result["tier_1"]
    assert "tier_2" in result
    assert "llm_evaluation" in result["tier_2"]
    assert "aggregation" in result
    assert result["aggregation"]["method"] == "spec_weighted_average"
    assert "disagreements" in result
    assert result["confidence_gate"]["decision"] in ("run_llm", "skip_llm")

def test_pipeline_without_llm_when_skipped():
    result = run_text_evaluation(
        user_answer="recursion uses base case and stack",
        reference_answer="recursion base case stack",
        llm_score=None, llm_max_score=30.0,
        llm_criteria_scores=None, llm_feedback=None, llm_model=None,
        semantic_normalized=0.95, max_score=30.0, pass_score=18.0, stakes="low",
    )
    assert result["tier_2"]["llm_evaluation"] is None
    assert result["confidence_gate"]["tier1_confidence"] > 0
