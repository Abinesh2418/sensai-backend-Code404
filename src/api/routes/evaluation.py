from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

from api.models import (
    Evaluation,
    EvaluationStatus,
    SubmitHumanFeedbackRequest,
    UpdateTrustWeightsRequest,
)
from api.db.evaluation import (
    get_evaluation,
    get_evaluation_for_user_task,
    get_pending_reviews,
    update_evaluation_status,
    store_evaluation_signal,
    get_trust_weights,
    set_trust_weights,
)
from api.utils.db import execute_db_operation
from api.evaluators.human_evaluator import HumanEvaluator
from api.evaluators.aggregator import aggregate_signals
from api.evaluators import EvaluationSignal as EvalSignal

router = APIRouter()

human_evaluator = HumanEvaluator()


# Static paths MUST come before parameterized paths to avoid FastAPI route conflicts

@router.get("/user/{user_id}/task/{task_id}", response_model=Evaluation)
async def get_user_task_evaluation(
    user_id: int,
    task_id: int,
    question_id: Optional[int] = None,
):
    evaluation = await get_evaluation_for_user_task(user_id, task_id, question_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation


@router.get("/review/pending")
async def get_pending_human_reviews(
    org_id: Optional[int] = None,
    cohort_id: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
):
    # If no org_id, fetch all provisional evaluations
    if org_id is None:
        from api.config import evaluations_table_name
        rows = await execute_db_operation(
            f"""SELECT e.id, e.user_id, e.task_id, e.question_id, e.status, e.final_score,
                       e.max_score, e.pass_score, e.explanation, e.created_at, e.updated_at
                FROM {evaluations_table_name} e
                WHERE e.status = 'provisional' AND e.deleted_at IS NULL
                ORDER BY e.created_at ASC LIMIT ? OFFSET ?""",
            (limit, offset),
            fetch_all=True,
        )
    else:
        evaluations = await get_pending_reviews(org_id, cohort_id, limit, offset)
        rows = None

    # Build enriched response for frontend
    results = []
    if rows is not None:
        from api.db.evaluation import _row_to_evaluation, get_signals_for_evaluation
        evaluations = []
        for row in rows:
            ev = _row_to_evaluation(row)
            ev.signals = await get_signals_for_evaluation(ev.id)
            evaluations.append(ev)

    for ev in evaluations:
        # Get user name
        user_row = await execute_db_operation(
            "SELECT first_name, last_name FROM users WHERE id = ?",
            (ev.user_id,), fetch_one=True,
        )
        user_name = " ".join(filter(None, [user_row[0] or "", user_row[1] or ""])).strip() if user_row else f"User {ev.user_id}"

        # Get task title
        task_row = await execute_db_operation(
            "SELECT title FROM tasks WHERE id = ?",
            (ev.task_id,), fetch_one=True,
        )
        task_title = task_row[0] if task_row else f"Task {ev.task_id}"

        # Get question title
        question_title = None
        if ev.question_id:
            q_row = await execute_db_operation(
                "SELECT title FROM questions WHERE id = ?",
                (ev.question_id,), fetch_one=True,
            )
            question_title = q_row[0] if q_row else None

        # Extract AI score from signals
        ai_score = None
        for sig in (ev.signals or []):
            if sig.evaluator_type == "ai" and sig.score is not None:
                ai_score = sig.score
                break

        results.append({
            "evaluation_id": ev.id,
            "user_id": ev.user_id,
            "user_name": user_name,
            "task_id": ev.task_id,
            "task_title": task_title,
            "question_id": ev.question_id,
            "question_title": question_title,
            "status": str(ev.status),
            "ai_score": ai_score,
            "max_score": ev.max_score,
            "created_at": str(ev.created_at),
        })

    return results


@router.post("/{evaluation_id}/review")
async def submit_human_feedback(
    evaluation_id: int,
    request: SubmitHumanFeedbackRequest,
):
    evaluation = await get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    if evaluation.status == EvaluationStatus.FINAL:
        raise HTTPException(status_code=400, detail="Evaluation is already finalized")

    # Create scored or feedback-only signal
    if request.criteria_scores:
        signal = human_evaluator.create_scored_signal(
            reviewer_user_id=request.reviewer_user_id,
            criteria_scores=request.criteria_scores,
            max_score_per_criterion=request.max_score_per_criterion or 10.0,
            overall_feedback=request.overall_feedback,
        )
    else:
        signal = human_evaluator.create_feedback_signal(
            reviewer_user_id=request.reviewer_user_id,
            overall_feedback=request.overall_feedback,
            criteria_feedback=request.criteria_feedback,
        )

    # Store the human signal
    await store_evaluation_signal(
        evaluation_id=evaluation_id,
        evaluator_type=signal.evaluator_type,
        evaluator_id=signal.evaluator_id,
        score=signal.score,
        max_score=signal.max_score,
        normalized_score=signal.normalized_score,
        confidence=signal.confidence,
        criteria_scores=signal.criteria_scores,
        feedback=signal.feedback,
        metadata=signal.metadata,
    )

    # Re-aggregate with spec weights (human gets 35% when scored)
    from api.engine.aggregator import aggregate_text_signals, aggregate_audio_signals

    # Extract existing signal normalized scores
    keyword_norm = None
    semantic_norm = None
    llm_norm = None
    fluency_norm = None
    confidence_norm = None
    is_audio = False
    for s in (evaluation.signals or []):
        if s.evaluator_type == "keyword":
            keyword_norm = s.normalized_score
        elif s.evaluator_type == "embedding":
            semantic_norm = s.normalized_score
        elif s.evaluator_type == "ai":
            llm_norm = s.normalized_score if s.max_score and s.score is not None else None
            if llm_norm is None and s.score is not None and s.max_score:
                llm_norm = s.score / s.max_score
        elif s.evaluator_type == "fluency":
            fluency_norm = s.normalized_score
            is_audio = True
        elif s.evaluator_type == "confidence":
            confidence_norm = s.normalized_score
            is_audio = True

    human_norm = signal.normalized_score

    if is_audio:
        aggregation = aggregate_audio_signals(
            fluency_normalized=fluency_norm,
            confidence_normalized=confidence_norm,
            semantic_normalized=semantic_norm,
            llm_normalized=llm_norm,
            human_normalized=human_norm,
            max_score=evaluation.max_score,
        )
    else:
        aggregation = aggregate_text_signals(
            keyword_normalized=keyword_norm,
            semantic_normalized=semantic_norm,
            llm_normalized=llm_norm,
            human_normalized=human_norm,
            max_score=evaluation.max_score,
        )

    final_score = aggregation["final_score"]

    await update_evaluation_status(
        evaluation_id=evaluation_id,
        status=EvaluationStatus.FINAL,
        final_score=final_score,
        explanation=aggregation,
    )

    return {"status": "ok", "evaluation_id": evaluation_id, "final_score": final_score}


@router.get("/{evaluation_id}/explain")
async def get_evaluation_explanation(evaluation_id: int):
    evaluation = await get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    return {
        "evaluation_id": evaluation.id,
        "status": str(evaluation.status),
        "final_score": evaluation.final_score,
        "max_score": evaluation.max_score,
        "pass_score": evaluation.pass_score,
        "passed": (
            evaluation.final_score >= evaluation.pass_score
            if evaluation.final_score is not None
            else None
        ),
        "explanation": evaluation.explanation,
    }


@router.get("/trust/weights")
async def get_org_trust_weights(org_id: int):
    weights = await get_trust_weights(org_id)
    return {"org_id": org_id, "weights": weights}


@router.put("/trust/weights")
async def update_org_trust_weights(org_id: int, request: UpdateTrustWeightsRequest):
    current = await get_trust_weights(org_id)

    if request.ai is not None:
        current["ai"] = request.ai
    if request.embedding is not None:
        current["embedding"] = request.embedding

    await set_trust_weights(org_id, current)
    return {"org_id": org_id, "weights": current}


async def _download_audio_bytes(uuid: str) -> bytes | None:
    """Download audio WAV bytes from S3 or local storage by UUID."""
    from api.settings import settings
    from api.utils.s3 import download_file_from_s3_as_bytes, get_media_upload_s3_key_from_uuid
    try:
        if settings.s3_folder_name:
            return download_file_from_s3_as_bytes(
                get_media_upload_s3_key_from_uuid(uuid, "wav")
            )
        else:
            path = os.path.join(settings.local_upload_folder, f"{uuid}.wav")
            with open(path, "rb") as f:
                return f.read()
    except Exception:
        return None


@router.post("/run-engine/{user_id}/{task_id}")
async def run_full_evaluation_engine(user_id: int, task_id: int):
    """
    Run ALL evaluation methods fresh and stream SSE events for each step
    so the frontend shows live per-question progress.
    """
    from fastapi.responses import StreamingResponse
    from api.config import evaluations_table_name, questions_table_name, chat_history_table_name
    from api.evaluators.embedding_evaluator import EmbeddingEvaluator
    from api.engine.llm_evaluator import run_llm_evaluation, run_audio_llm_evaluation
    from api.db.evaluation import store_evaluation_signal, update_evaluation_status, create_evaluation
    from api.db.task import get_question
    from api.db.utils import construct_description_from_blocks
    from api.engine import run_text_evaluation, run_audio_evaluation
    from api.engine.keyword_evaluator import evaluate_keywords
    from api.engine.confidence_gate import (
        compute_tier1_weighted_score, should_run_llm,
        compute_audio_tier1_confidence, should_run_llm_audio,
    )
    from api.engine.transcriber import transcribe_audio
    from api.engine.fluency_evaluator import evaluate_fluency
    from api.engine.confidence_evaluator import evaluate_confidence
    import json, os

    # Pre-fetch all data before streaming (errors here → normal HTTP errors)
    task_row = await execute_db_operation(
        "SELECT id, title, org_id FROM tasks WHERE id = ? AND deleted_at IS NULL",
        (task_id,), fetch_one=True,
    )
    if not task_row:
        raise HTTPException(status_code=404, detail="Task not found")
    task_title = task_row[1]

    q_rows = await execute_db_operation(
        f"SELECT id, title, answer, blocks, input_type FROM {questions_table_name} WHERE task_id = ? AND deleted_at IS NULL ORDER BY position",
        (task_id,), fetch_all=True,
    )
    if not q_rows:
        raise HTTPException(status_code=404, detail="No questions found")

    user_row = await execute_db_operation(
        "SELECT first_name, last_name FROM users WHERE id = ?", (user_id,), fetch_one=True,
    )
    user_name = " ".join(filter(None, [user_row[0] or "", user_row[1] or ""])).strip() if user_row else f"User {user_id}"

    chat_rows = await execute_db_operation(
        f"SELECT question_id, content FROM {chat_history_table_name} WHERE user_id = ? AND question_id IN ({','.join(str(r[0]) for r in q_rows)}) AND role = 'user' AND deleted_at IS NULL",
        (user_id,), fetch_all=True,
    )
    user_answers = {r[0]: r[1] for r in (chat_rows or [])}

    eval_rows = await execute_db_operation(
        f"SELECT id, question_id, max_score, pass_score FROM {evaluations_table_name} WHERE user_id = ? AND task_id = ? AND deleted_at IS NULL",
        (user_id, task_id), fetch_all=True,
    )
    eval_map = {r[1]: {"id": r[0], "max_score": r[2], "pass_score": r[3]} for r in eval_rows}

    async def generate():
        embedding_evaluator = EmbeddingEvaluator()
        questions_analysis = []
        total_final_score = 0.0
        total_max_score = 0.0
        all_disagreements = []

        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        for idx, q_row in enumerate(q_rows):
            q_id, q_title, q_answer_raw, q_blocks_raw = q_row[0], q_row[1], q_row[2], q_row[3]
            q_input_type = q_row[4] if len(q_row) > 4 else None
            user_answer = user_answers.get(q_id, "")

            if not user_answer:
                q_data = {"question_id": q_id, "question_title": q_title, "status": "no_answer", "tiers": {}}
                questions_analysis.append(q_data)
                yield sse({"event": "question_done", "q_idx": idx, "result": q_data})
                continue

            ev = eval_map.get(q_id)
            if not ev:
                eval_id = await create_evaluation(user_id=user_id, task_id=task_id, max_score=30.0, pass_score=18.0, question_id=q_id)
                ev = {"id": eval_id, "max_score": 30.0, "pass_score": 18.0}
                eval_map[q_id] = ev
            evaluation_id = ev["id"]
            await update_evaluation_status(evaluation_id, EvaluationStatus.IN_PROGRESS)

            reference_text = ""
            if q_answer_raw:
                try:
                    reference_text = construct_description_from_blocks(json.loads(q_answer_raw))
                except Exception:
                    reference_text = str(q_answer_raw)

            question_text = ""
            if q_blocks_raw:
                try:
                    question_text = construct_description_from_blocks(json.loads(q_blocks_raw))
                except Exception:
                    question_text = q_title

            question_data_row = await get_question(q_id)
            scorecard = question_data_row.get("scorecard") if question_data_row else None

            # ---- AUDIO PIPELINE ----
            if str(q_input_type).lower() == "audio":
                # Download audio bytes (user_answer is the audio UUID)
                yield sse({"event": "step", "q_idx": idx, "q_title": q_title, "step": "download_audio", "status": "running"})
                audio_data = await _download_audio_bytes(user_answer)
                if not audio_data:
                    q_data = {"question_id": q_id, "question_title": q_title, "status": "error", "error": "Audio not found", "tiers": {}}
                    questions_analysis.append(q_data)
                    yield sse({"event": "question_done", "q_idx": idx, "result": q_data})
                    continue
                yield sse({"event": "step", "q_idx": idx, "step": "download_audio", "status": "done", "summary": f"{len(audio_data)} bytes"})

                # Transcription
                yield sse({"event": "step", "q_idx": idx, "step": "transcription", "status": "running"})
                transcript_result = transcribe_audio(audio_data)
                yield sse({"event": "step", "q_idx": idx, "step": "transcription", "status": "done",
                           "summary": f"{transcript_result['word_count']} words, {transcript_result['duration_seconds']:.0f}s"})

                # Fluency (Tier-1)
                yield sse({"event": "step", "q_idx": idx, "step": "fluency", "status": "running"})
                fluency_result = evaluate_fluency(audio_data, transcript_result, ev["max_score"])
                await store_evaluation_signal(
                    evaluation_id=evaluation_id, evaluator_type="fluency",
                    evaluator_id="local", score=fluency_result["score"],
                    max_score=fluency_result["max_score"], normalized_score=fluency_result["normalized_score"],
                    confidence=fluency_result["confidence"], weight=0.10,
                    feedback=f"WPM: {fluency_result['words_per_minute']:.0f}, Fillers: {fluency_result['filler_count']}",
                    metadata={"filler_ratio": fluency_result["filler_ratio"], "silence_ratio": fluency_result["silence_ratio"],
                              "words_per_minute": fluency_result["words_per_minute"], "long_pauses": fluency_result["long_pauses"],
                              "deductions": fluency_result["deductions"]},
                )
                yield sse({"event": "step", "q_idx": idx, "step": "fluency", "status": "done",
                           "summary": f"{fluency_result['words_per_minute']:.0f} WPM, {fluency_result['filler_count']} fillers, score {fluency_result['score']:.1f}/{fluency_result['max_score']}"})

                # Confidence (Tier-1)
                yield sse({"event": "step", "q_idx": idx, "step": "confidence_detection", "status": "running"})
                confidence_det_result = evaluate_confidence(audio_data, transcript_result, ev["max_score"])
                await store_evaluation_signal(
                    evaluation_id=evaluation_id, evaluator_type="confidence",
                    evaluator_id="local", score=confidence_det_result["score"],
                    max_score=confidence_det_result["max_score"], normalized_score=confidence_det_result["normalized_score"],
                    confidence=confidence_det_result["confidence"], weight=0.10,
                    feedback=f"Stability: {confidence_det_result['voice_stability']:.0%}, Hedging: {confidence_det_result['hedging_phrases_count']}",
                    metadata={"hedging_ratio": confidence_det_result["hedging_ratio"],
                              "false_starts_per_minute": confidence_det_result["false_starts_per_minute"],
                              "voice_stability": confidence_det_result["voice_stability"],
                              "deductions": confidence_det_result["deductions"]},
                )
                yield sse({"event": "step", "q_idx": idx, "step": "confidence_detection", "status": "done",
                           "summary": f"stability {confidence_det_result['voice_stability']:.0%}, score {confidence_det_result['score']:.1f}/{confidence_det_result['max_score']}"})

                # Semantic Similarity on transcript (Tier-1)
                semantic_result = None
                semantic_norm = None
                yield sse({"event": "step", "q_idx": idx, "step": "semantic", "status": "running"})
                if transcript_result["word_count"] >= 5 and reference_text:
                    try:
                        sem_signal = await embedding_evaluator.evaluate(
                            submission_content=transcript_result["transcript"],
                            context={"reference_answers": [reference_text], "max_score": ev["max_score"]},
                        )
                        if sem_signal:
                            semantic_norm = sem_signal.normalized_score
                            semantic_result = {
                                "score": sem_signal.score, "max_score": sem_signal.max_score,
                                "normalized_score": sem_signal.normalized_score,
                                "confidence": sem_signal.confidence, "model": sem_signal.evaluator_id,
                                "feedback": sem_signal.feedback,
                                "cosine_similarity": (sem_signal.metadata or {}).get("cosine_similarity"),
                            }
                            await store_evaluation_signal(
                                evaluation_id=evaluation_id, evaluator_type="embedding",
                                evaluator_id=sem_signal.evaluator_id, score=sem_signal.score,
                                max_score=sem_signal.max_score, normalized_score=sem_signal.normalized_score,
                                confidence=sem_signal.confidence, weight=0.20,
                                criteria_scores=None, feedback=sem_signal.feedback, metadata=sem_signal.metadata,
                            )
                            yield sse({"event": "step", "q_idx": idx, "step": "semantic", "status": "done",
                                       "summary": f"{semantic_norm*100:.1f}% similarity"})
                        else:
                            yield sse({"event": "step", "q_idx": idx, "step": "semantic", "status": "skipped", "summary": "no signal"})
                    except Exception:
                        yield sse({"event": "step", "q_idx": idx, "step": "semantic", "status": "skipped", "summary": "error"})
                else:
                    reason = "transcript too short" if transcript_result["word_count"] < 5 else "no reference"
                    yield sse({"event": "step", "q_idx": idx, "step": "semantic", "status": "skipped", "summary": reason})

                # Audio Confidence Gate
                tier1_conf = compute_audio_tier1_confidence(
                    fluency_result["normalized_score"], confidence_det_result["normalized_score"], semantic_norm,
                )
                llm_needed = should_run_llm_audio(
                    fluency_result["normalized_score"], confidence_det_result["normalized_score"], semantic_norm,
                )
                yield sse({"event": "step", "q_idx": idx, "step": "confidence_gate", "status": "done",
                           "summary": f"Confidence {tier1_conf*100:.0f}% → {'LLM needed' if llm_needed else 'LLM skipped (≥85%)'}"})

                tier1_context = (
                    f"- Fluency: {fluency_result['normalized_score']*100:.0f}% (WPM: {fluency_result['words_per_minute']:.0f}, fillers: {fluency_result['filler_count']})\n"
                    f"- Confidence: {confidence_det_result['normalized_score']*100:.0f}% (stability: {confidence_det_result['voice_stability']:.0%})"
                )
                if semantic_norm is not None:
                    tier1_context += f"\n- Semantic similarity: {semantic_norm*100:.1f}%"

                # Audio LLM Evaluation (Tier-2)
                llm_result = None
                llm_score = None
                llm_max = None
                if llm_needed:
                    yield sse({"event": "step", "q_idx": idx, "step": "llm", "status": "running"})
                    llm_result = await run_audio_llm_evaluation(
                        transcript=transcript_result["transcript"],
                        question_text=question_text or q_title,
                        reference_answer=reference_text,
                        scorecard=scorecard,
                        tier1_context=tier1_context,
                        user_name=user_name,
                    )
                    if llm_result:
                        llm_score = llm_result["score"]
                        llm_max = llm_result["max_score"]
                        await store_evaluation_signal(
                            evaluation_id=evaluation_id, evaluator_type="ai",
                            evaluator_id=llm_result["model"], score=llm_result["score"],
                            max_score=llm_result["max_score"], normalized_score=llm_result["normalized_score"],
                            confidence=llm_result["confidence"], weight=0.60,
                            criteria_scores=llm_result.get("criteria_scores"), feedback=llm_result.get("feedback"),
                            metadata={"model": llm_result["model"], "chain_of_thought": llm_result.get("chain_of_thought")},
                        )
                        yield sse({"event": "step", "q_idx": idx, "step": "llm", "status": "done",
                                   "summary": f"{llm_result['score']:.1f}/{llm_result['max_score']} ({llm_result['normalized_score']*100:.0f}%)"})
                    else:
                        yield sse({"event": "step", "q_idx": idx, "step": "llm", "status": "skipped", "summary": "no result"})
                else:
                    yield sse({"event": "step", "q_idx": idx, "step": "llm", "status": "skipped",
                               "summary": f"Tier-1 sufficient ({tier1_conf*100:.0f}% ≥ 85%)"})

                # Aggregate + persist (audio)
                engine_result = run_audio_evaluation(
                    transcript_result=transcript_result, reference_answer=reference_text,
                    fluency_result=fluency_result, confidence_result=confidence_det_result,
                    llm_score=llm_score, llm_max_score=llm_max or ev["max_score"],
                    llm_criteria_scores=llm_result["criteria_scores"] if llm_result else None,
                    llm_feedback=llm_result["feedback"] if llm_result else None,
                    llm_model=llm_result["model"] if llm_result else None,
                    semantic_normalized=semantic_norm,
                    max_score=ev["max_score"], pass_score=ev["pass_score"],
                )
                final_score = engine_result["final_score"]
                await update_evaluation_status(
                    evaluation_id=evaluation_id, status=EvaluationStatus.FINAL,
                    final_score=final_score, explanation=engine_result["aggregation"],
                )

                total_final_score += final_score
                total_max_score += ev["max_score"]
                for d in engine_result["disagreements"]:
                    all_disagreements.append({**d, "question_id": q_id, "question_title": q_title})

                tier_1 = engine_result["tier_1"]
                if semantic_result and not semantic_result.get("error"):
                    tier_1["semantic_similarity"] = semantic_result

                q_data = {
                    "question_id": q_id, "question_title": q_title,
                    "evaluation_id": evaluation_id, "status": "final",
                    "final_score": final_score, "max_score": ev["max_score"],
                    "pass_score": ev["pass_score"], "passed": engine_result["passed"],
                    "tiers": {"tier_1": tier_1, "tier_2": engine_result["tier_2"]},
                    "confidence_gate": engine_result["confidence_gate"],
                    "aggregation": engine_result["aggregation"],
                    "disagreements": engine_result["disagreements"],
                }
                questions_analysis.append(q_data)
                yield sse({"event": "question_done", "q_idx": idx, "result": q_data})
                continue

            # ---- TEXT PIPELINE (existing, unchanged) ----

            # STEP 1: Keyword Matching
            yield sse({"event": "step", "q_idx": idx, "q_title": q_title, "step": "keyword", "status": "running"})
            keyword_result = evaluate_keywords(user_answer, reference_text, ev["max_score"])
            yield sse({"event": "step", "q_idx": idx, "step": "keyword", "status": "done",
                       "summary": f"{keyword_result['matched_count']}/{keyword_result['total_reference_keywords']} keywords ({keyword_result['coverage']*100:.0f}%)"})

            # STEP 2: Semantic Similarity
            yield sse({"event": "step", "q_idx": idx, "step": "semantic", "status": "running"})
            semantic_result = None
            semantic_norm = None
            if reference_text:
                try:
                    sem_signal = await embedding_evaluator.evaluate(
                        submission_content=user_answer,
                        context={"reference_answers": [reference_text], "max_score": ev["max_score"]},
                    )
                    if sem_signal:
                        semantic_norm = sem_signal.normalized_score
                        semantic_result = {
                            "score": sem_signal.score, "max_score": sem_signal.max_score,
                            "normalized_score": sem_signal.normalized_score,
                            "confidence": sem_signal.confidence, "model": sem_signal.evaluator_id,
                            "feedback": sem_signal.feedback,
                            "cosine_similarity": (sem_signal.metadata or {}).get("cosine_similarity"),
                        }
                        await store_evaluation_signal(
                            evaluation_id=evaluation_id, evaluator_type="embedding",
                            evaluator_id=sem_signal.evaluator_id, score=sem_signal.score,
                            max_score=sem_signal.max_score, normalized_score=sem_signal.normalized_score,
                            confidence=sem_signal.confidence, weight=sem_signal.confidence,
                            criteria_scores=None, feedback=sem_signal.feedback, metadata=sem_signal.metadata,
                        )
                        yield sse({"event": "step", "q_idx": idx, "step": "semantic", "status": "done",
                                   "summary": f"{semantic_norm*100:.1f}% similarity"})
                    else:
                        yield sse({"event": "step", "q_idx": idx, "step": "semantic", "status": "skipped", "summary": "no signal"})
                except Exception as e:
                    semantic_result = {"error": str(e)}
                    yield sse({"event": "step", "q_idx": idx, "step": "semantic", "status": "skipped", "summary": "error"})
            else:
                yield sse({"event": "step", "q_idx": idx, "step": "semantic", "status": "skipped", "summary": "no reference"})

            # STEP 3: Confidence Gate
            tier1_score = compute_tier1_weighted_score(keyword_result["normalized_score"], semantic_norm)
            llm_needed = should_run_llm(keyword_result["normalized_score"], semantic_norm)
            yield sse({"event": "step", "q_idx": idx, "step": "confidence_gate", "status": "done",
                       "summary": f"Score {tier1_score*100:.0f}% → {'LLM needed' if llm_needed else 'LLM skipped (≥85%)'}"})

            tier1_context = f"- Keyword coverage: {keyword_result['coverage']*100:.0f}% ({keyword_result['matched_count']}/{keyword_result['total_reference_keywords']} keywords)"
            if semantic_norm is not None:
                tier1_context += f"\n- Semantic similarity: {semantic_norm*100:.1f}%"

            # STEP 4: LLM Evaluation
            llm_result = None
            llm_score = None
            llm_max = None
            if llm_needed:
                yield sse({"event": "step", "q_idx": idx, "step": "llm", "status": "running"})
                llm_result = await run_llm_evaluation(
                    user_answer=user_answer,
                    question_text=question_text or q_title,
                    reference_answer=reference_text,
                    scorecard=scorecard,
                    tier1_context=tier1_context,
                    user_name=user_name,
                )
                if llm_result:
                    llm_score = llm_result["score"]
                    llm_max = llm_result["max_score"]
                    await store_evaluation_signal(
                        evaluation_id=evaluation_id, evaluator_type="ai",
                        evaluator_id=llm_result["model"], score=llm_result["score"],
                        max_score=llm_result["max_score"], normalized_score=llm_result["normalized_score"],
                        confidence=llm_result["confidence"], weight=0.60,
                        criteria_scores=llm_result["criteria_scores"], feedback=llm_result["feedback"],
                        metadata={"model": llm_result["model"], "chain_of_thought": llm_result.get("chain_of_thought")},
                    )
                    yield sse({"event": "step", "q_idx": idx, "step": "llm", "status": "done",
                               "summary": f"{llm_result['score']:.1f}/{llm_result['max_score']} ({llm_result['normalized_score']*100:.0f}%)"})
                else:
                    yield sse({"event": "step", "q_idx": idx, "step": "llm", "status": "skipped", "summary": "no result"})
            else:
                yield sse({"event": "step", "q_idx": idx, "step": "llm", "status": "skipped",
                           "summary": f"Tier-1 sufficient ({tier1_score*100:.0f}% ≥ 85%)"})

            # Aggregate + persist
            engine_result = run_text_evaluation(
                user_answer=user_answer, reference_answer=reference_text,
                llm_score=llm_score, llm_max_score=llm_max or ev["max_score"],
                llm_criteria_scores=llm_result["criteria_scores"] if llm_result else None,
                llm_feedback=llm_result["feedback"] if llm_result else None,
                llm_model=llm_result["model"] if llm_result else None,
                semantic_normalized=semantic_norm,
                max_score=ev["max_score"], pass_score=ev["pass_score"],
            )
            final_score = engine_result["final_score"]
            await update_evaluation_status(
                evaluation_id=evaluation_id, status=EvaluationStatus.FINAL,
                final_score=final_score, explanation=engine_result["aggregation"],
            )

            total_final_score += final_score
            total_max_score += ev["max_score"]
            for d in engine_result["disagreements"]:
                all_disagreements.append({**d, "question_id": q_id, "question_title": q_title})

            tier_1 = engine_result["tier_1"]
            if semantic_result and not semantic_result.get("error"):
                tier_1["semantic_similarity"] = semantic_result

            q_data = {
                "question_id": q_id, "question_title": q_title,
                "evaluation_id": evaluation_id, "status": "final",
                "final_score": final_score, "max_score": ev["max_score"],
                "pass_score": ev["pass_score"], "passed": engine_result["passed"],
                "tiers": {"tier_1": tier_1, "tier_2": engine_result["tier_2"]},
                "confidence_gate": engine_result["confidence_gate"],
                "aggregation": engine_result["aggregation"],
                "disagreements": engine_result["disagreements"],
            }
            questions_analysis.append(q_data)
            yield sse({"event": "question_done", "q_idx": idx, "result": q_data})

        overall_pct = round((total_final_score / total_max_score) * 100, 1) if total_max_score > 0 else 0
        yield sse({"event": "complete", "result": {
            "user_id": user_id, "user_name": user_name,
            "task_id": task_id, "task_title": task_title,
            "overall": {
                "final_score": round(total_final_score, 2),
                "max_score": total_max_score,
                "percentage": overall_pct,
                "questions_evaluated": len([q for q in questions_analysis if q.get("status") == "final"]),
                "total_questions": len(q_rows),
            },
            "disagreements": all_disagreements,
            "questions": questions_analysis,
        }})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Parameterized path — must be last to avoid catching static routes
@router.get("/{evaluation_id}", response_model=Evaluation)
async def get_evaluation_by_id(evaluation_id: int):
    evaluation = await get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation
