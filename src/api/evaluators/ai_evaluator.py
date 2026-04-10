"""
AI Evaluator — extracts an EvaluationSignal from the existing LLM output.

This evaluator does NOT make a separate LLM call. It takes the already-produced
AI response (from /ai/chat or /ai/assignment) and converts it into a standardized
EvaluationSignal for the aggregation engine.
"""

from typing import Optional, Dict
from api.evaluators import BaseEvaluator, EvaluationSignal
from api.config import openai_plan_to_model_name


class AIEvaluator(BaseEvaluator):
    """Extracts evaluation signal from existing AI/LLM output."""

    async def evaluate(
        self,
        submission_content: str,
        scorecard: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> Optional[EvaluationSignal]:
        # This method is not used directly — use extract_signal_from_* methods instead
        raise NotImplementedError(
            "Use extract_signal_from_subjective() or extract_signal_from_assignment()"
        )

    def extract_signal_from_subjective(
        self,
        llm_output: Dict,
        scorecard: Dict,
        model_name: Optional[str] = None,
    ) -> Optional[EvaluationSignal]:
        """
        Extract signal from subjective question LLM output.

        llm_output shape (from routes/ai.py):
        {
            "chain_of_thought": "...",
            "feedback": "...",
            "scorecard": {
                "CriterionName": {
                    "feedback": {"correct": "...", "wrong": "..."},
                    "score": 20,
                    "max_score": 25,
                    "pass_score": 15
                },
                ...
            }
        }
        """
        if not llm_output:
            return None

        llm_scorecard = llm_output.get("scorecard")
        if not llm_scorecard:
            # No scorecard in output means student's response wasn't a valid answer
            return None

        # Calculate total score from per-criterion scores
        total_score = 0.0
        total_max = 0.0
        criteria_scores = {}

        for criterion_name, criterion_data in llm_scorecard.items():
            if not isinstance(criterion_data, dict):
                continue

            score = criterion_data.get("score", 0)
            max_score = criterion_data.get("max_score", 0)

            total_score += score
            total_max += max_score

            criteria_scores[criterion_name] = {
                "score": score,
                "max_score": max_score,
                "pass_score": criterion_data.get("pass_score", 0),
                "feedback": criterion_data.get("feedback", {}),
            }

        normalized = total_score / total_max if total_max > 0 else 0.0

        return EvaluationSignal(
            evaluator_type="ai",
            evaluator_id=model_name or openai_plan_to_model_name.get("text", "unknown"),
            score=total_score,
            max_score=total_max,
            normalized_score=normalized,
            confidence=0.85,
            criteria_scores=criteria_scores,
            feedback=llm_output.get("feedback"),
            metadata={
                "model": model_name or openai_plan_to_model_name.get("text"),
                "chain_of_thought": llm_output.get("chain_of_thought"),
            },
        )

    def extract_signal_from_objective(
        self,
        llm_output: Dict,
        max_score: float,
        model_name: Optional[str] = None,
    ) -> Optional[EvaluationSignal]:
        """
        Extract signal from objective question LLM output.

        llm_output shape:
        {
            "analysis": "...",
            "feedback": "...",
            "is_correct": true/false
        }
        """
        if not llm_output:
            return None

        is_correct = llm_output.get("is_correct", False)
        score = max_score if is_correct else 0.0
        normalized = 1.0 if is_correct else 0.0

        return EvaluationSignal(
            evaluator_type="ai",
            evaluator_id=model_name or openai_plan_to_model_name.get("text", "unknown"),
            score=score,
            max_score=max_score,
            normalized_score=normalized,
            confidence=0.90,
            feedback=llm_output.get("feedback"),
            metadata={
                "model": model_name or openai_plan_to_model_name.get("text"),
                "is_correct": is_correct,
                "analysis": llm_output.get("analysis"),
            },
        )

    def extract_signal_from_assignment(
        self,
        llm_output: Dict,
        scorecard: Dict,
        evaluation_criteria: Dict,
        model_name: Optional[str] = None,
    ) -> Optional[EvaluationSignal]:
        """
        Extract signal from assignment LLM output.

        llm_output shape (from routes/ai.py):
        {
            "chain_of_thought": "...",
            "feedback": "...",
            "evaluation_status": "in_progress" | "needs_resubmission" | "completed",
            "key_area_scores": {
                "CriterionName": {
                    "feedback": {"correct": "...", "wrong": "..."},
                    "score": 40,
                    "max_score": 50,
                    "pass_score": 30
                },
                ...
            },
            "assignment_score": 78.0  (only for file submissions)
        }
        """
        if not llm_output:
            return None

        evaluation_status = llm_output.get("evaluation_status")

        # Only extract a meaningful signal when evaluation is complete or has scores
        key_area_scores = llm_output.get("key_area_scores", {})
        assignment_score = llm_output.get("assignment_score")

        max_score = evaluation_criteria.get("max_score", 100)

        # Calculate score from key_area_scores if available
        if key_area_scores:
            total_score = 0.0
            total_max = 0.0
            criteria_scores = {}

            for area_name, area_data in key_area_scores.items():
                if not isinstance(area_data, dict):
                    continue

                score = area_data.get("score", 0)
                area_max = area_data.get("max_score", 0)

                total_score += score
                total_max += area_max

                criteria_scores[area_name] = {
                    "score": score,
                    "max_score": area_max,
                    "pass_score": area_data.get("pass_score", 0),
                    "feedback": area_data.get("feedback", {}),
                }

            normalized = total_score / total_max if total_max > 0 else 0.0
            # Scale to assignment's max_score
            final_score = normalized * max_score
        elif assignment_score is not None:
            final_score = assignment_score
            normalized = final_score / max_score if max_score > 0 else 0.0
            criteria_scores = {}
        else:
            # No scores available yet (evaluation still in progress)
            return None

        # Reasoning model gets higher confidence
        confidence = 0.90 if "o3" in (model_name or "") else 0.85

        return EvaluationSignal(
            evaluator_type="ai",
            evaluator_id=model_name or openai_plan_to_model_name.get("reasoning", "unknown"),
            score=final_score,
            max_score=max_score,
            normalized_score=normalized,
            confidence=confidence,
            criteria_scores=criteria_scores,
            feedback=llm_output.get("feedback"),
            metadata={
                "model": model_name or openai_plan_to_model_name.get("reasoning"),
                "evaluation_status": evaluation_status,
                "chain_of_thought": llm_output.get("chain_of_thought"),
            },
        )
