from typing import Optional
from api.engine.classifier import classify_submission
from api.engine.keyword_evaluator import evaluate_keywords
from api.engine.confidence_gate import (
    compute_tier1_weighted_score, should_run_llm, should_queue_human_review,
    compute_audio_tier1_confidence, should_run_llm_audio, should_queue_human_review_audio,
)
from api.engine.aggregator import aggregate_text_signals, aggregate_audio_signals
from api.engine.disagreement import detect_disagreements

def run_text_evaluation(
    user_answer: str, reference_answer: str,
    llm_score: Optional[float], llm_max_score: float,
    llm_criteria_scores: Optional[dict], llm_feedback: Optional[str],
    llm_model: Optional[str], semantic_normalized: Optional[float],
    max_score: float, pass_score: float,
    human_normalized: Optional[float] = None,
) -> dict:
    submission_type = classify_submission(input_type="text", response_type="chat")

    # TIER 1: Keyword Matching
    keyword_result = evaluate_keywords(user_answer, reference_answer, max_score)
    keyword_normalized = keyword_result["normalized_score"]

    # CONFIDENCE GATE — decision based purely on weighted Tier-1 score
    tier1_weighted = compute_tier1_weighted_score(keyword_normalized, semantic_normalized)
    llm_needed = should_run_llm(keyword_normalized, semantic_normalized)
    human_needed = should_queue_human_review(keyword_normalized, semantic_normalized)

    # TIER 2: LLM (already computed, decide whether to include in aggregation)
    llm_normalized = None
    llm_detail = None
    if llm_score is not None and llm_max_score > 0:
        llm_normalized = llm_score / llm_max_score
        llm_detail = {
            "score": llm_score, "max_score": llm_max_score,
            "normalized_score": round(llm_normalized, 4),
            "model": llm_model, "feedback": llm_feedback,
            "criteria_scores": llm_criteria_scores,
        }

    llm_normalized_for_agg = llm_normalized if llm_needed else None

    # AGGREGATE
    aggregation = aggregate_text_signals(
        keyword_normalized=keyword_normalized, semantic_normalized=semantic_normalized,
        llm_normalized=llm_normalized_for_agg, human_normalized=human_normalized,
        max_score=max_score,
    )

    # DISAGREEMENT DETECTION
    score_map = {"keyword": keyword_normalized, "semantic": semantic_normalized}
    if llm_normalized_for_agg is not None:
        score_map["llm"] = llm_normalized_for_agg
    if human_normalized is not None:
        score_map["human"] = human_normalized
    disagreements = detect_disagreements(score_map)

    final_score = aggregation["final_score"]

    return {
        "submission_type": submission_type,
        "final_score": final_score,
        "max_score": max_score,
        "pass_score": pass_score,
        "passed": final_score >= pass_score,
        "tier_1": {
            "keyword_matching": keyword_result,
            "semantic_similarity": {"normalized_score": semantic_normalized} if semantic_normalized is not None else None,
        },
        "confidence_gate": {
            "tier1_weighted_score": round(tier1_weighted, 4),
            "decision": "skip_llm" if not llm_needed else "run_llm",
            "human_review_recommended": human_needed,
        },
        "tier_2": {
            "llm_evaluation": llm_detail,
            "included_in_aggregation": llm_normalized_for_agg is not None,
        },
        "aggregation": aggregation,
        "disagreements": disagreements,
    }


def run_audio_evaluation(
    transcript_result: dict,
    reference_answer: str,
    fluency_result: dict,
    confidence_result: dict,
    llm_score: Optional[float],
    llm_max_score: float,
    llm_criteria_scores: Optional[dict],
    llm_feedback: Optional[str],
    llm_model: Optional[str],
    semantic_normalized: Optional[float],
    max_score: float,
    pass_score: float,
    human_normalized: Optional[float] = None,
) -> dict:
    """
    Orchestrate audio evaluation: Tier-1 → confidence gate → Tier-2 → aggregation → disagreement.
    Mirrors run_text_evaluation in structure.
    """
    submission_type = classify_submission(input_type="audio")

    fluency_normalized = fluency_result.get("normalized_score")
    confidence_normalized = confidence_result.get("normalized_score")

    # CONFIDENCE GATE
    tier1_confidence = compute_audio_tier1_confidence(
        fluency_normalized, confidence_normalized, semantic_normalized,
    )
    llm_needed = should_run_llm_audio(fluency_normalized, confidence_normalized, semantic_normalized)
    human_needed = should_queue_human_review_audio(fluency_normalized, confidence_normalized, semantic_normalized)

    # TIER 2: LLM (already computed, decide whether to include)
    llm_normalized = None
    llm_detail = None
    if llm_score is not None and llm_max_score > 0:
        llm_normalized = llm_score / llm_max_score
        llm_detail = {
            "score": llm_score, "max_score": llm_max_score,
            "normalized_score": round(llm_normalized, 4),
            "model": llm_model, "feedback": llm_feedback,
            "criteria_scores": llm_criteria_scores,
        }

    llm_normalized_for_agg = llm_normalized if llm_needed else None

    # AGGREGATE
    aggregation = aggregate_audio_signals(
        fluency_normalized=fluency_normalized,
        confidence_normalized=confidence_normalized,
        semantic_normalized=semantic_normalized,
        llm_normalized=llm_normalized_for_agg,
        human_normalized=human_normalized,
        max_score=max_score,
    )

    # DISAGREEMENT DETECTION
    score_map = {
        "fluency": fluency_normalized,
        "confidence": confidence_normalized,
        "semantic": semantic_normalized,
    }
    if llm_normalized_for_agg is not None:
        score_map["llm"] = llm_normalized_for_agg
    if human_normalized is not None:
        score_map["human"] = human_normalized
    disagreements = detect_disagreements(score_map)

    final_score = aggregation["final_score"]

    return {
        "submission_type": submission_type,
        "final_score": final_score,
        "max_score": max_score,
        "pass_score": pass_score,
        "passed": final_score >= pass_score,
        "tier_1": {
            "fluency": fluency_result,
            "confidence": confidence_result,
            "semantic_similarity": {"normalized_score": semantic_normalized} if semantic_normalized is not None else None,
            "transcript": transcript_result.get("transcript", ""),
        },
        "confidence_gate": {
            "tier1_confidence": round(tier1_confidence, 4),
            "decision": "skip_llm" if not llm_needed else "run_llm",
            "human_review_recommended": human_needed,
        },
        "tier_2": {
            "llm_evaluation": llm_detail,
            "included_in_aggregation": llm_normalized_for_agg is not None,
        },
        "aggregation": aggregation,
        "disagreements": disagreements,
    }
