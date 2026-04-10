"""
Evaluation Pipeline — orchestrates multi-signal evaluation after AI response.

Called after the existing AI endpoints finish producing their response.
Runs the embedding evaluator async and computes provisional scores.
Does NOT block or delay the streaming AI response to students.
"""

import asyncio
from typing import Optional, Dict, List

from api.evaluators.ai_evaluator import AIEvaluator
from api.evaluators.embedding_evaluator import EmbeddingEvaluator
from api.evaluators.aggregator import aggregate_signals, update_trust_on_new_signals
from api.evaluators import EvaluationSignal
from api.db.evaluation import (
    create_evaluation,
    store_evaluation_signal,
    update_evaluation_status,
    get_trust_weights,
    get_signals_for_evaluation,
)
from api.models import EvaluationStatus
from api.utils.logging import logger

ai_evaluator = AIEvaluator()
embedding_evaluator = EmbeddingEvaluator()


async def run_evaluation_pipeline(
    user_id: int,
    task_id: int,
    org_id: int,
    llm_output: Dict,
    submission_content: str,
    max_score: float,
    pass_score: float,
    question_id: Optional[int] = None,
    question_type: Optional[str] = None,
    scorecard: Optional[Dict] = None,
    evaluation_criteria: Optional[Dict] = None,
    reference_answers: Optional[List[str]] = None,
    model_name: Optional[str] = None,
):
    """
    Run the full evaluation pipeline after AI response completes.

    This is meant to be called via asyncio.create_task() so it doesn't
    block the streaming response to the student.
    """
    try:
        # 1. Create evaluation record
        evaluation_id = await create_evaluation(
            user_id=user_id,
            task_id=task_id,
            max_score=max_score,
            pass_score=pass_score,
            question_id=question_id,
        )

        await update_evaluation_status(
            evaluation_id, EvaluationStatus.IN_PROGRESS
        )

        # 2. Extract AI signal from the already-produced LLM output
        ai_signal = _extract_ai_signal(
            llm_output=llm_output,
            question_type=question_type,
            scorecard=scorecard,
            evaluation_criteria=evaluation_criteria,
            max_score=max_score,
            model_name=model_name,
        )

        if ai_signal:
            await _store_signal(evaluation_id, ai_signal)

        # 3. Run embedding evaluator
        embedding_signal = await embedding_evaluator.evaluate(
            submission_content=submission_content,
            context={
                "reference_answers": reference_answers or [],
                "max_score": max_score,
            },
        )

        if embedding_signal:
            await _store_signal(evaluation_id, embedding_signal)

        # 4. Update trust weights based on inter-signal agreement
        if ai_signal and embedding_signal:
            await update_trust_on_new_signals(org_id, evaluation_id)

        # 5. Compute provisional score
        trust_weights = await get_trust_weights(org_id)

        signals = []
        if ai_signal:
            signals.append(ai_signal)
        if embedding_signal:
            signals.append(embedding_signal)

        if signals:
            final_score, explanation = aggregate_signals(
                signals, trust_weights, max_score
            )

            await update_evaluation_status(
                evaluation_id=evaluation_id,
                status=EvaluationStatus.PROVISIONAL,
                final_score=final_score,
                explanation=explanation,
            )
        else:
            await update_evaluation_status(
                evaluation_id, EvaluationStatus.PROVISIONAL
            )

        logger.info(
            f"Evaluation pipeline completed: eval_id={evaluation_id}, "
            f"ai={'yes' if ai_signal else 'no'}, "
            f"embedding={'yes' if embedding_signal else 'no'}"
        )

        return evaluation_id

    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
        return None


def _extract_ai_signal(
    llm_output: Dict,
    question_type: Optional[str],
    scorecard: Optional[Dict],
    evaluation_criteria: Optional[Dict],
    max_score: float,
    model_name: Optional[str],
) -> Optional[EvaluationSignal]:
    """Extract AI signal based on question/task type."""
    if not llm_output:
        return None

    if question_type == "objective":
        return ai_evaluator.extract_signal_from_objective(
            llm_output, max_score, model_name
        )
    elif question_type == "subjective":
        return ai_evaluator.extract_signal_from_subjective(
            llm_output, scorecard or {}, model_name
        )
    elif question_type == "assignment":
        return ai_evaluator.extract_signal_from_assignment(
            llm_output, scorecard or {}, evaluation_criteria or {}, model_name
        )

    return None


async def _store_signal(evaluation_id: int, signal: EvaluationSignal):
    """Store an EvaluationSignal to the database."""
    await store_evaluation_signal(
        evaluation_id=evaluation_id,
        evaluator_type=signal.evaluator_type,
        evaluator_id=signal.evaluator_id,
        score=signal.score,
        max_score=signal.max_score,
        normalized_score=signal.normalized_score,
        confidence=signal.confidence,
        weight=signal.confidence,  # Weight at time of scoring
        criteria_scores=signal.criteria_scores,
        feedback=signal.feedback,
        metadata=signal.metadata,
    )
