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


def test_full_audio_pipeline_all_tiers():
    from api.engine import run_audio_evaluation

    fluency_result = {
        "score": 8.0, "max_score": 10.0, "normalized_score": 0.80,
        "confidence": 0.90, "filler_count": 2, "filler_ratio": 0.04,
        "silence_ratio": 0.10, "words_per_minute": 130.0, "long_pauses": 0,
        "duration_seconds": 30.0, "deductions": {}, "filler_words_found": ["um"],
    }
    confidence_result = {
        "score": 7.0, "max_score": 10.0, "normalized_score": 0.70,
        "confidence": 0.90, "hedging_phrases_count": 1, "hedging_ratio": 0.05,
        "volume_drop_count": 0, "false_starts": 1, "false_starts_per_minute": 2.0,
        "voice_stability": 0.85, "strong_delivery_bonus": False, "deductions": {},
    }
    transcript_result = {
        "transcript": "Recursion uses a base case to stop",
        "words": [], "duration_seconds": 30.0, "word_count": 7,
    }

    result = run_audio_evaluation(
        transcript_result=transcript_result,
        reference_answer="Recursion is a technique with base case and recursive case",
        fluency_result=fluency_result,
        confidence_result=confidence_result,
        llm_score=25.0, llm_max_score=30.0,
        llm_criteria_scores={"Content": {"score": 8, "max_score": 10}},
        llm_feedback="Good answer", llm_model="gpt-4o-audio",
        semantic_normalized=0.75,
        max_score=30.0, pass_score=18.0,
    )

    assert result["submission_type"] == "audio"
    assert result["final_score"] > 0
    assert result["max_score"] == 30.0
    assert "tier_1" in result
    assert "fluency" in result["tier_1"]
    assert "confidence" in result["tier_1"]
    assert "semantic_similarity" in result["tier_1"]
    assert "transcript" in result["tier_1"]
    assert "tier_2" in result
    assert "aggregation" in result
    assert result["aggregation"]["method"] == "spec_weighted_average"
    assert "disagreements" in result
    assert result["confidence_gate"]["decision"] in ("run_llm", "skip_llm")


def test_audio_pipeline_without_llm():
    from api.engine import run_audio_evaluation

    fluency_result = {"normalized_score": 0.95, "score": 9.5, "max_score": 10.0}
    confidence_result = {"normalized_score": 0.90, "score": 9.0, "max_score": 10.0}
    transcript_result = {"transcript": "test", "words": [], "duration_seconds": 30.0, "word_count": 1}

    result = run_audio_evaluation(
        transcript_result=transcript_result,
        reference_answer="test reference",
        fluency_result=fluency_result,
        confidence_result=confidence_result,
        llm_score=None, llm_max_score=30.0,
        llm_criteria_scores=None, llm_feedback=None, llm_model=None,
        semantic_normalized=0.92,
        max_score=30.0, pass_score=18.0,
    )

    assert result["submission_type"] == "audio"
    assert result["tier_2"]["llm_evaluation"] is None
    assert result["final_score"] > 0
