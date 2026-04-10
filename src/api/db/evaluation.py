import json
from typing import List, Dict, Optional

from api.config import (
    evaluations_table_name,
    evaluation_signals_table_name,
    evaluator_trust_table_name,
    DEFAULT_TRUST_WEIGHTS,
)
from api.utils.db import get_new_db_connection, execute_db_operation
from api.models import (
    Evaluation,
    EvaluationSignalModel,
    EvaluationStatus,
    EvaluatorType,
    EvaluatorTrust,
)


# --- Evaluation CRUD ---


async def create_evaluation(
    user_id: int,
    task_id: int,
    max_score: float,
    pass_score: float,
    question_id: Optional[int] = None,
) -> int:
    return await execute_db_operation(
        f"""INSERT INTO {evaluations_table_name}
            (user_id, task_id, question_id, status, max_score, pass_score)
            VALUES (?, ?, ?, ?, ?, ?)""",
        (user_id, task_id, question_id, str(EvaluationStatus.PENDING), max_score, pass_score),
        get_last_row_id=True,
    )


async def get_evaluation(evaluation_id: int) -> Optional[Evaluation]:
    row = await execute_db_operation(
        f"""SELECT id, user_id, task_id, question_id, status, final_score,
                   max_score, pass_score, explanation, created_at, updated_at
            FROM {evaluations_table_name}
            WHERE id = ? AND deleted_at IS NULL""",
        (evaluation_id,),
        fetch_one=True,
    )

    if not row:
        return None

    evaluation = _row_to_evaluation(row)

    # Attach signals
    evaluation.signals = await get_signals_for_evaluation(evaluation_id)

    return evaluation


async def get_evaluation_for_user_task(
    user_id: int,
    task_id: int,
    question_id: Optional[int] = None,
) -> Optional[Evaluation]:
    if question_id:
        row = await execute_db_operation(
            f"""SELECT id, user_id, task_id, question_id, status, final_score,
                       max_score, pass_score, explanation, created_at, updated_at
                FROM {evaluations_table_name}
                WHERE user_id = ? AND task_id = ? AND question_id = ? AND deleted_at IS NULL
                ORDER BY created_at DESC LIMIT 1""",
            (user_id, task_id, question_id),
            fetch_one=True,
        )
    else:
        row = await execute_db_operation(
            f"""SELECT id, user_id, task_id, question_id, status, final_score,
                       max_score, pass_score, explanation, created_at, updated_at
                FROM {evaluations_table_name}
                WHERE user_id = ? AND task_id = ? AND question_id IS NULL AND deleted_at IS NULL
                ORDER BY created_at DESC LIMIT 1""",
            (user_id, task_id),
            fetch_one=True,
        )

    if not row:
        return None

    evaluation = _row_to_evaluation(row)
    evaluation.signals = await get_signals_for_evaluation(evaluation.id)

    return evaluation


async def update_evaluation_status(
    evaluation_id: int,
    status: EvaluationStatus,
    final_score: Optional[float] = None,
    explanation: Optional[Dict] = None,
):
    explanation_json = json.dumps(explanation) if explanation else None

    await execute_db_operation(
        f"""UPDATE {evaluations_table_name}
            SET status = ?, final_score = ?, explanation = ?
            WHERE id = ? AND deleted_at IS NULL""",
        (str(status), final_score, explanation_json, evaluation_id),
    )


# --- Evaluation Signals ---


async def store_evaluation_signal(
    evaluation_id: int,
    evaluator_type: str,
    evaluator_id: Optional[str] = None,
    score: Optional[float] = None,
    max_score: Optional[float] = None,
    normalized_score: Optional[float] = None,
    confidence: Optional[float] = None,
    weight: Optional[float] = None,
    criteria_scores: Optional[Dict] = None,
    feedback: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> int:
    criteria_json = json.dumps(criteria_scores) if criteria_scores else None
    metadata_json = json.dumps(metadata) if metadata else None

    return await execute_db_operation(
        f"""INSERT INTO {evaluation_signals_table_name}
            (evaluation_id, evaluator_type, evaluator_id, score, max_score,
             normalized_score, confidence, weight, criteria_scores, feedback, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            evaluation_id, evaluator_type, evaluator_id, score, max_score,
            normalized_score, confidence, weight, criteria_json, feedback, metadata_json,
        ),
        get_last_row_id=True,
    )


async def get_signals_for_evaluation(evaluation_id: int) -> List[EvaluationSignalModel]:
    rows = await execute_db_operation(
        f"""SELECT id, evaluation_id, evaluator_type, evaluator_id, score, max_score,
                   normalized_score, confidence, weight, criteria_scores, feedback,
                   metadata, created_at
            FROM {evaluation_signals_table_name}
            WHERE evaluation_id = ? AND deleted_at IS NULL
            ORDER BY created_at ASC""",
        (evaluation_id,),
        fetch_all=True,
    )

    if not rows:
        return []

    return [_row_to_signal(row) for row in rows]


async def get_signal_by_type(
    evaluation_id: int, evaluator_type: str
) -> Optional[EvaluationSignalModel]:
    row = await execute_db_operation(
        f"""SELECT id, evaluation_id, evaluator_type, evaluator_id, score, max_score,
                   normalized_score, confidence, weight, criteria_scores, feedback,
                   metadata, created_at
            FROM {evaluation_signals_table_name}
            WHERE evaluation_id = ? AND evaluator_type = ? AND deleted_at IS NULL
            ORDER BY created_at DESC LIMIT 1""",
        (evaluation_id, evaluator_type),
        fetch_one=True,
    )

    if not row:
        return None

    return _row_to_signal(row)


# --- Pending Reviews ---


async def get_pending_reviews(
    org_id: int,
    cohort_id: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[Evaluation]:
    # Evaluations in provisional state (AI + embedding done, awaiting human feedback)
    query = f"""
        SELECT e.id, e.user_id, e.task_id, e.question_id, e.status, e.final_score,
               e.max_score, e.pass_score, e.explanation, e.created_at, e.updated_at
        FROM {evaluations_table_name} e
        JOIN tasks t ON e.task_id = t.id
        WHERE t.org_id = ?
          AND e.status = ?
          AND e.deleted_at IS NULL
        ORDER BY e.created_at ASC
        LIMIT ? OFFSET ?
    """
    params = (org_id, str(EvaluationStatus.PROVISIONAL), limit, offset)

    rows = await execute_db_operation(query, params, fetch_all=True)

    if not rows:
        return []

    evaluations = []
    for row in rows:
        evaluation = _row_to_evaluation(row)
        evaluation.signals = await get_signals_for_evaluation(evaluation.id)
        evaluations.append(evaluation)

    return evaluations


# --- Trust Weights ---


async def get_or_create_trust(org_id: int, evaluator_type: str) -> EvaluatorTrust:
    row = await execute_db_operation(
        f"""SELECT id, org_id, evaluator_type, trust_weight,
                   total_agreements, total_evaluations, updated_at
            FROM {evaluator_trust_table_name}
            WHERE org_id = ? AND evaluator_type = ?""",
        (org_id, evaluator_type),
        fetch_one=True,
    )

    if row:
        return EvaluatorTrust(
            id=row[0],
            org_id=row[1],
            evaluator_type=EvaluatorType(row[2]),
            trust_weight=row[3],
            total_agreements=row[4],
            total_evaluations=row[5],
            updated_at=row[6],
        )

    # Create with default weight
    default_weight = DEFAULT_TRUST_WEIGHTS.get(evaluator_type, 0.5)

    new_id = await execute_db_operation(
        f"""INSERT INTO {evaluator_trust_table_name}
            (org_id, evaluator_type, trust_weight)
            VALUES (?, ?, ?)""",
        (org_id, evaluator_type, default_weight),
        get_last_row_id=True,
    )

    return EvaluatorTrust(
        id=new_id,
        org_id=org_id,
        evaluator_type=EvaluatorType(evaluator_type),
        trust_weight=default_weight,
    )


async def get_trust_weights(org_id: int) -> Dict[str, float]:
    rows = await execute_db_operation(
        f"""SELECT evaluator_type, trust_weight
            FROM {evaluator_trust_table_name}
            WHERE org_id = ?""",
        (org_id,),
        fetch_all=True,
    )

    weights = dict(DEFAULT_TRUST_WEIGHTS)

    if rows:
        for row in rows:
            weights[row[0]] = row[1]

    return weights


async def update_trust(
    org_id: int,
    evaluator_type: str,
    total_agreements: float,
    total_evaluations: int,
    trust_weight: float,
):
    await execute_db_operation(
        f"""UPDATE {evaluator_trust_table_name}
            SET trust_weight = ?, total_agreements = ?, total_evaluations = ?
            WHERE org_id = ? AND evaluator_type = ?""",
        (trust_weight, total_agreements, total_evaluations, org_id, evaluator_type),
    )


async def set_trust_weights(org_id: int, weights: Dict[str, float]):
    for evaluator_type, weight in weights.items():
        trust = await get_or_create_trust(org_id, evaluator_type)
        await execute_db_operation(
            f"""UPDATE {evaluator_trust_table_name}
                SET trust_weight = ?
                WHERE org_id = ? AND evaluator_type = ?""",
            (weight, org_id, evaluator_type),
        )


# --- Row Converters ---


def _row_to_evaluation(row) -> Evaluation:
    explanation = None
    if row[8]:
        try:
            explanation = json.loads(row[8])
        except (json.JSONDecodeError, TypeError):
            explanation = None

    return Evaluation(
        id=row[0],
        user_id=row[1],
        task_id=row[2],
        question_id=row[3],
        status=EvaluationStatus(row[4]),
        final_score=row[5],
        max_score=row[6],
        pass_score=row[7],
        explanation=explanation,
        created_at=row[9],
        updated_at=row[10],
    )


def _row_to_signal(row) -> EvaluationSignalModel:
    criteria_scores = None
    if row[9]:
        try:
            criteria_scores = json.loads(row[9])
        except (json.JSONDecodeError, TypeError):
            criteria_scores = None

    metadata = None
    if row[11]:
        try:
            metadata = json.loads(row[11])
        except (json.JSONDecodeError, TypeError):
            metadata = None

    return EvaluationSignalModel(
        id=row[0],
        evaluation_id=row[1],
        evaluator_type=EvaluatorType(row[2]),
        evaluator_id=row[3],
        score=row[4],
        max_score=row[5],
        normalized_score=row[6],
        confidence=row[7],
        weight=row[8],
        criteria_scores=criteria_scores,
        feedback=row[10],
        metadata=metadata,
        created_at=row[12],
    )
