from api.engine.weights import get_weights_for_type


def aggregate_text_signals(
    keyword_normalized: float | None,
    semantic_normalized: float | None,
    llm_normalized: float | None,
    human_normalized: float | None,
    max_score: float,
) -> dict:
    has_human = human_normalized is not None
    base_weights = get_weights_for_type("text", has_human=has_human)

    # Map evaluator names to their normalized scores
    signal_map = {
        "keyword": keyword_normalized,
        "semantic": semantic_normalized,
        "llm": llm_normalized,
    }
    if has_human:
        signal_map["human"] = human_normalized

    # Filter to active signals (not None)
    active_signals = {k: v for k, v in signal_map.items() if v is not None}

    # Sum of base weights for active signals
    active_base_sum = sum(base_weights[k] for k in active_signals)

    # Redistribute weights proportionally
    redistributed = {
        k: base_weights[k] / active_base_sum
        for k in active_signals
    }

    # Compute final normalized score
    final_normalized = sum(
        score * redistributed[k] for k, score in active_signals.items()
    )

    final_score = final_normalized * max_score

    # Build per-signal details list
    signals = [
        {
            "name": k,
            "normalized_score": score,
            "weight": redistributed[k],
            "contribution": score * redistributed[k],
        }
        for k, score in active_signals.items()
    ]

    return {
        "final_score": final_score,
        "max_score": max_score,
        "final_normalized": final_normalized,
        "weights_used": redistributed,
        "method": "spec_weighted_average",
        "has_human_review": has_human,
        "signals": signals,
    }


def aggregate_audio_signals(
    fluency_normalized: float | None,
    confidence_normalized: float | None,
    semantic_normalized: float | None,
    llm_normalized: float | None,
    human_normalized: float | None,
    max_score: float,
) -> dict:
    """
    Aggregate audio evaluation signals using AUDIO_WEIGHTS from weights.py.
    Same proportional redistribution pattern as aggregate_text_signals.
    """
    has_human = human_normalized is not None
    base_weights = get_weights_for_type("audio", has_human=has_human)

    signal_map = {
        "fluency": fluency_normalized,
        "confidence": confidence_normalized,
        "semantic": semantic_normalized,
        "llm": llm_normalized,
    }
    if has_human:
        signal_map["human"] = human_normalized

    active_signals = {k: v for k, v in signal_map.items() if v is not None}

    if not active_signals:
        return {
            "final_score": 0.0, "max_score": max_score, "final_normalized": 0.0,
            "weights_used": {}, "method": "spec_weighted_average",
            "has_human_review": has_human, "signals": [],
        }

    active_base_sum = sum(base_weights[k] for k in active_signals)
    redistributed = {k: base_weights[k] / active_base_sum for k in active_signals}

    final_normalized = sum(score * redistributed[k] for k, score in active_signals.items())
    final_score = final_normalized * max_score

    signals = [
        {
            "name": k,
            "normalized_score": score,
            "weight": redistributed[k],
            "contribution": score * redistributed[k],
        }
        for k, score in active_signals.items()
    ]

    return {
        "final_score": final_score,
        "max_score": max_score,
        "final_normalized": final_normalized,
        "weights_used": redistributed,
        "method": "spec_weighted_average",
        "has_human_review": has_human,
        "signals": signals,
    }
