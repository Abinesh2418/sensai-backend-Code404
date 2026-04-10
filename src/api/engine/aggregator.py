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
