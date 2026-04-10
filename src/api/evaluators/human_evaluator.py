"""
Human Evaluator — supports both feedback-only and scored reviews.

Mentors can provide qualitative feedback (feedback-only) or numerical
per-criterion scores that contribute 35% weight to the final evaluation.
"""

from typing import Optional, Dict
from api.evaluators import BaseEvaluator, EvaluationSignal


class HumanEvaluator(BaseEvaluator):
    """Handles human mentor feedback and scored reviews."""

    async def evaluate(
        self,
        submission_content: str,
        scorecard: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> Optional[EvaluationSignal]:
        raise NotImplementedError("Use create_feedback_signal() or create_scored_signal()")

    def create_feedback_signal(
        self,
        reviewer_user_id: int,
        overall_feedback: str,
        criteria_feedback: Optional[Dict[str, str]] = None,
    ) -> EvaluationSignal:
        """Create a feedback-only signal (no numerical score)."""
        return EvaluationSignal(
            evaluator_type="human",
            evaluator_id=str(reviewer_user_id),
            score=None,
            max_score=None,
            normalized_score=None,
            confidence=None,
            criteria_scores=criteria_feedback,
            feedback=overall_feedback,
            metadata={"reviewer_user_id": reviewer_user_id},
        )

    def create_scored_signal(
        self,
        reviewer_user_id: int,
        criteria_scores: Dict[str, float],
        max_score_per_criterion: float,
        overall_feedback: str,
    ) -> EvaluationSignal:
        """
        Create a scored signal from a mentor's review with per-criterion scores.

        criteria_scores shape: {"Understanding": 8.0, "Application": 7.0, "Clarity": 9.0}
        Each score is out of max_score_per_criterion (e.g. 10).

        The human signal contributes 35% weight to the final evaluation
        when re-aggregated through the spec engine.
        """
        total_score = sum(criteria_scores.values())
        total_max = max_score_per_criterion * len(criteria_scores)
        normalized = total_score / total_max if total_max > 0 else 0.0

        return EvaluationSignal(
            evaluator_type="human",
            evaluator_id=str(reviewer_user_id),
            score=total_score,
            max_score=total_max,
            normalized_score=normalized,
            confidence=1.0,
            criteria_scores={
                k: {"score": v, "max_score": max_score_per_criterion}
                for k, v in criteria_scores.items()
            },
            feedback=overall_feedback,
            metadata={"reviewer_user_id": reviewer_user_id, "scored": True},
        )
