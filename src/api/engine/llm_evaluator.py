"""
Tier-2 LLM Evaluator — makes a real OpenAI API call to evaluate a text submission.

Uses the same prompt structure as routes/ai.py for subjective questions,
but runs synchronously (non-streaming) for the evaluation engine.
"""

from typing import Optional
from pydantic import BaseModel, Field, create_model
from api.llm import run_llm_with_openai
from api.config import openai_plan_to_model_name
from api.prompts import compile_prompt
from api.prompts.subjective_question import (
    SUBJECTIVE_QUESTION_SYSTEM_PROMPT,
    SUBJECTIVE_QUESTION_USER_PROMPT,
)
from api.utils.logging import logger


class _Feedback(BaseModel):
    correct: Optional[str] = Field(description="What worked well")
    wrong: Optional[str] = Field(description="What needs improvement")


class _Row(BaseModel):
    feedback: _Feedback
    score: float = Field(description="Score within the min/max range")
    max_score: float = Field(description="Maximum score for this criterion")
    pass_score: float = Field(description="Pass score for this criterion")


def _make_scorecard_model(criteria_names: list[str]):
    field_defs = {name: (_Row, ...) for name in criteria_names}
    return create_model("Scorecard", **field_defs)


class _BaseOutput(BaseModel):
    chain_of_thought: str = Field(description="Concise analysis of the student's response")
    feedback: str = Field(description="Comprehensive summary for the student")


def _convert_scorecard_to_prompt(scorecard: dict) -> str:
    parts = []
    for i, c in enumerate(scorecard.get("criteria", [])):
        name = c["name"].replace('"', "")
        parts.append(
            f'Criterion {i + 1}:\n**Name**: **{name}** '
            f'[min_score: {c["min_score"]}, max_score: {c["max_score"]}, '
            f'pass_score: {c.get("pass_score", c["max_score"])}]\n\n{c["description"]}'
        )
    return "\n\n".join(parts)


async def run_llm_evaluation(
    user_answer: str,
    question_text: str,
    reference_answer: str,
    scorecard: Optional[dict],
    tier1_context: Optional[str] = None,
    user_name: Optional[str] = None,
) -> dict:
    """
    Run a fresh LLM evaluation for a text submission.

    Returns dict with: score, max_score, normalized_score, confidence,
    criteria_scores, feedback, model, chain_of_thought.
    """
    # Build question details
    question_details = f"**Task**\n\n{question_text}\n\n"

    if scorecard and scorecard.get("criteria"):
        scorecard_prompt = _convert_scorecard_to_prompt(scorecard)
        question_details += f"---\n\n**Scoring Criteria**\n\n{scorecard_prompt}\n\n"

    # Add Tier-1 context for more informed LLM evaluation
    if tier1_context:
        question_details += f"---\n\n**Additional evaluation context (for your reference):**\n{tier1_context}\n\n"

    user_details = f"Student name: {user_name}" if user_name else "Student name: Unknown"

    messages = compile_prompt(
        SUBJECTIVE_QUESTION_SYSTEM_PROMPT,
        SUBJECTIVE_QUESTION_USER_PROMPT,
        task_details=question_details,
        user_details=user_details,
    )

    # Add the student's answer as a user message
    messages.append({"role": "user", "content": user_answer})

    # Build dynamic output model with scorecard criteria
    if scorecard and scorecard.get("criteria"):
        criteria_names = [c["name"].replace('"', "") for c in scorecard["criteria"]]
        ScorecardModel = _make_scorecard_model(criteria_names)

        class Output(_BaseOutput):
            scorecard: Optional[ScorecardModel] = Field(
                description="Score and feedback for each criterion"
            )
    else:
        class Output(_BaseOutput):
            scorecard: Optional[dict] = None

    # Pick model
    model = openai_plan_to_model_name["text"]

    try:
        result = await run_llm_with_openai(
            model=model,
            messages=messages,
            response_model=Output,
            max_output_tokens=4096,
            api_mode="responses",
        )

        llm_output = result.model_dump() if hasattr(result, "model_dump") else result

        # Extract scores
        llm_scorecard = llm_output.get("scorecard") or {}
        total_score = 0.0
        total_max = 0.0
        criteria_scores = {}

        for crit_name, crit_data in llm_scorecard.items():
            if not isinstance(crit_data, dict):
                continue
            s = crit_data.get("score", 0)
            m = crit_data.get("max_score", 0)
            total_score += s
            total_max += m
            criteria_scores[crit_name] = {
                "score": s,
                "max_score": m,
                "pass_score": crit_data.get("pass_score", 0),
                "feedback": crit_data.get("feedback", {}),
            }

        normalized = total_score / total_max if total_max > 0 else 0.0

        return {
            "score": total_score,
            "max_score": total_max,
            "normalized_score": round(normalized, 4),
            "confidence": 0.85,
            "criteria_scores": criteria_scores,
            "feedback": llm_output.get("feedback"),
            "model": model,
            "chain_of_thought": llm_output.get("chain_of_thought"),
        }

    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}", exc_info=True)
        return None
