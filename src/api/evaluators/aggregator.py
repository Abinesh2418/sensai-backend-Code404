"""
Aggregation Engine — combines evaluation signals into a final score with explanation.

All three signal types (AI, embedding, human) contribute to the final score.
Human signals participate in scoring when a mentor provides criterion-level scores.
The evaluation only reaches "final" status after mentor review is complete.

Trust calibration is based on inter-signal agreement between AI and embedding.
"""

from typing import List, Dict, Optional, Tuple
from api.evaluators import EvaluationSignal
from api.config import DEFAULT_TRUST_WEIGHTS
from api.db.evaluation import (
    get_or_create_trust,
    update_trust,
    get_signal_by_type,
)

TRUST_WEIGHT_FLOOR = 0.10
ALL_EVALUATOR_TYPES = {"ai", "embedding", "human"}


def aggregate_signals(
    signals: List[EvaluationSignal],
    trust_weights: Dict[str, float],
    max_score: float,
) -> Tuple[float, Dict]:
    """
    Compute weighted final score from all available signals.

    Returns (final_score, explanation_json).
    Human signals contribute to the score when they have normalized_score.
    If human has no numerical score, their qualitative feedback is still included.
    """
    scoring_signals = [
        s for s in signals
        if s.normalized_score is not None
    ]

    if not scoring_signals:
        return 0.0, _build_explanation(signals, trust_weights, 0.0, max_score)

    total_weighted = 0.0
    total_effective_weight = 0.0

    signal_details = []

    for signal in scoring_signals:
        weight = trust_weights.get(signal.evaluator_type, 0.5)
        confidence = signal.confidence or 1.0
        effective_weight = weight * confidence

        total_weighted += signal.normalized_score * effective_weight
        total_effective_weight += effective_weight

        signal_details.append({
            "type": signal.evaluator_type,
            "evaluator_id": signal.evaluator_id,
            "raw_score": signal.score,
            "max_score": signal.max_score,
            "normalized_score": round(signal.normalized_score, 4),
            "weight": round(weight, 4),
            "confidence": round(confidence, 4),
            "effective_weight": round(effective_weight, 4),
            "contributes_to_score": True,
        })

    final_normalized = total_weighted / total_effective_weight if total_effective_weight > 0 else 0.0
    final_score = round(final_normalized * max_score, 2)

    # Add contribution percentages
    for detail in signal_details:
        contribution = (
            detail["effective_weight"] * detail["normalized_score"]
        ) / total_weighted if total_weighted > 0 else 0.0
        detail["contribution_pct"] = f"{contribution * 100:.1f}%"

    # Add non-scoring signals (human feedback without scores) as feedback-only
    scored_types = {s.evaluator_type for s in scoring_signals}
    for signal in signals:
        if signal.evaluator_type not in scored_types:
            signal_details.append({
                "type": signal.evaluator_type,
                "evaluator_id": signal.evaluator_id,
                "score": None,
                "feedback": signal.feedback,
                "criteria_feedback": signal.criteria_scores,
                "contributes_to_score": False,
            })

    explanation = {
        "signals": signal_details,
        "method": "weighted_average_with_dynamic_trust",
        "total_effective_weight": round(total_effective_weight, 4),
        "final_normalized": round(final_normalized, 4),
        "has_human_review": any(s.evaluator_type == "human" for s in signals),
    }

    return final_score, explanation


def _build_explanation(
    signals: List[EvaluationSignal],
    trust_weights: Dict[str, float],
    final_score: float,
    max_score: float,
) -> Dict:
    """Build explanation when no scoring signals are available."""
    signal_details = []
    for signal in signals:
        signal_details.append({
            "type": signal.evaluator_type,
            "evaluator_id": signal.evaluator_id,
            "feedback": signal.feedback,
            "contributes_to_score": False,
        })

    return {
        "signals": signal_details,
        "method": "weighted_average_with_dynamic_trust",
        "total_effective_weight": 0,
        "final_normalized": 0,
        "note": "No scoring signals available",
    }


async def update_trust_on_new_signals(org_id: int, evaluation_id: int):
    """
    Update trust weights based on inter-signal agreement between AI and embedding.

    Called after both automated signals are stored for an evaluation.
    Trust calibration uses agreement between AI and embedding as the signal.
    """
    ai_signal = await get_signal_by_type(evaluation_id, "ai")
    embedding_signal = await get_signal_by_type(evaluation_id, "embedding")

    if not ai_signal or not embedding_signal:
        return

    if ai_signal.normalized_score is None or embedding_signal.normalized_score is None:
        return

    # Agreement = 1 - absolute deviation (0 = complete disagreement, 1 = perfect agreement)
    agreement = 1.0 - abs(ai_signal.normalized_score - embedding_signal.normalized_score)

    for evaluator_type in ["ai", "embedding"]:
        trust = await get_or_create_trust(org_id, evaluator_type)

        new_agreements = trust.total_agreements + agreement
        new_evaluations = trust.total_evaluations + 1

        base_weight = DEFAULT_TRUST_WEIGHTS.get(evaluator_type, 0.5)
        raw_weight = base_weight * (new_agreements / new_evaluations)
        new_weight = max(TRUST_WEIGHT_FLOOR, raw_weight)

        await update_trust(
            org_id=org_id,
            evaluator_type=evaluator_type,
            total_agreements=new_agreements,
            total_evaluations=new_evaluations,
            trust_weight=new_weight,
        )
