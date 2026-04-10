# Multi-Modal Evaluation Engine — Design Spec

## Context

SensAI currently evaluates student submissions (text, code, essays) through a single AI evaluator (OpenAI LLM with rubric-based scorecards). This creates several problems:

- **Single point of failure**: One model's bias or error directly becomes the student's score
- **No fairness guarantees**: A response that's semantically correct but stylistically different may be unfairly scored
- **No human oversight**: Mentors have no mechanism to review, override, or contribute scores
- **No explainability**: Students see feedback but not how their final score was derived

This engine introduces **multi-signal evaluation** — aggregating AI, embedding-based semantic matching, and human mentor review into a single fair, explainable assessment.

---

## Architecture: Pipeline with Progressive Scoring

```
Student submits text/essay response
        |
        v
  [Evaluation record created] (status: pending)
        |
        +---> [AI Evaluator]        --> EvaluationSignal (immediate)
        +---> [Embedding Evaluator] --> EvaluationSignal (immediate, ~200ms)
        +---> [Human Review Queue]  --> EvaluationSignal (async, when mentor reviews)
        |
        v  (after each new signal arrives)
  [Aggregator]
        |
        v
  evaluations.final_score = weighted combination
  evaluations.explanation = JSON breakdown of each signal's contribution
  evaluations.status = pending -> provisional -> final
```

**Progressive scoring**: AI + embedding signals produce a provisional score immediately. When a mentor reviews, the score updates to final. Students are notified of score changes.

---

## Data Model

### Table: `evaluations`

Tracks overall evaluation state per submission.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| user_id | INTEGER FK → users | Who submitted |
| task_id | INTEGER FK → tasks | Which task |
| question_id | INTEGER FK → questions (nullable) | Which question (quizzes) |
| status | TEXT | `pending` / `in_progress` / `provisional` / `final` |
| final_score | REAL (nullable) | Aggregated weighted score |
| max_score | REAL | From scorecard/evaluation_criteria |
| pass_score | REAL | From scorecard/evaluation_criteria |
| explanation | TEXT (nullable) | JSON: score derivation breakdown |
| created_at | TEXT | Timestamp |
| updated_at | TEXT | Timestamp |
| deleted_at | TEXT (nullable) | Soft delete |

**Indexes**: `(user_id, task_id)`, `(user_id, question_id)`, `(status)`

### Table: `evaluation_signals`

One row per evaluator per evaluation.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| evaluation_id | INTEGER FK → evaluations | Parent evaluation |
| evaluator_type | TEXT | `ai` / `embedding` / `human` |
| evaluator_id | TEXT (nullable) | Model name (e.g., "gpt-4.1") or mentor user_id |
| score | REAL | Raw score from evaluator |
| max_score | REAL | Scale this evaluator used |
| normalized_score | REAL | Score normalized to 0.0–1.0 |
| confidence | REAL | 0.0–1.0, evaluator's certainty |
| weight | REAL | Trust weight at time of scoring |
| criteria_scores | TEXT (nullable) | JSON: per-criterion breakdown |
| feedback | TEXT (nullable) | Evaluator's textual feedback |
| metadata | TEXT (nullable) | JSON: model version, embedding model, similarity score, etc. |
| created_at | TEXT | Timestamp |
| updated_at | TEXT | Timestamp |
| deleted_at | TEXT (nullable) | Soft delete |

**Indexes**: `(evaluation_id)`, `(evaluator_type)`

### Table: `evaluator_trust`

Dynamic trust weights per organization.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| org_id | INTEGER FK → organizations | Organization |
| evaluator_type | TEXT | `ai` / `embedding` / `human` |
| trust_weight | REAL | Current weight |
| total_agreements | REAL | Cumulative agreement score |
| total_evaluations | INTEGER | Total signals from this evaluator |
| created_at | TEXT | Timestamp |
| updated_at | TEXT | Timestamp |

**Default weights**: ai=0.65, embedding=0.35 (human is feedback-only, no weight in score calculation)
**Unique constraint**: `(org_id, evaluator_type)`

---

## Evaluator Interface

All evaluators implement a common contract:

```python
class EvaluationSignal(BaseModel):
    evaluator_type: str          # "ai", "embedding", "human"
    evaluator_id: Optional[str]  # Model name or user ID
    score: float                 # Raw score
    max_score: float             # Max possible on evaluator's scale
    normalized_score: float      # 0.0-1.0
    confidence: float            # 0.0-1.0
    criteria_scores: Optional[dict]  # Per-criterion breakdown
    feedback: Optional[str]      # Textual feedback
    metadata: Optional[dict]     # Extra info (model version, etc.)

class BaseEvaluator(ABC):
    @abstractmethod
    async def evaluate(
        self,
        submission_content: str,
        scorecard: Scorecard,
        context: dict
    ) -> EvaluationSignal:
        ...
```

### AI Evaluator

- **Refactored from**: existing logic in `src/api/routes/ai.py`
- Uses OpenAI LLM with scorecard-based prompt (existing prompt templates)
- Returns structured per-criterion scores and feedback
- Confidence: derived from model's chain-of-thought certainty (0.7-0.95 range)
- **No behavior change for students** — the AI still streams feedback as before

### Embedding Evaluator

- **New module**: `src/api/evaluators/embedding_evaluator.py`
- Uses OpenAI `text-embedding-3-small` model
- Flow:
  1. Embed student response
  2. Embed reference answer(s) from question's `answer` field
  3. Compute cosine similarity
  4. Map similarity to score range: `score = similarity * max_score`
- Confidence: scales with number of reference answers (1 ref = 0.5, 3+ refs = 0.8)
- Metadata includes: embedding model, similarity score, reference count

### Human Evaluator (Feedback-Only)

- **New workflow**: mentor reviews via API endpoint
- Mentor sees: student submission, AI feedback, embedding score (advisory)
- Mentor provides: **qualitative feedback only** — per-criterion textual comments and overall feedback. No numerical scores.
- Human feedback does NOT contribute to the weighted score calculation
- Purpose: provide richer, nuanced guidance that AI cannot (mentorship, encouragement, domain insights)
- Stored as evaluation_signal with `evaluator_type="human"`, `evaluator_id` = mentor's user_id, `score=None`, `feedback` populated

---

## Aggregation Engine

### Weighted Average Formula (Automated Signals Only)

The numerical score is computed from **AI and embedding signals only**. Human feedback is qualitative and does not enter the formula.

```
final_score = (Σ signal_i.normalized_score × weight_i × confidence_i) / (Σ weight_i × confidence_i) × max_score
```

Where `signal_i` iterates over automated evaluators (AI, embedding) only. `weight_i` comes from `evaluator_trust` for the org, and `confidence_i` from the signal itself.

### Progressive Scoring States

| State | When | Signals Present |
|-------|------|-----------------|
| `pending` | Evaluation created | None yet |
| `in_progress` | At least one signal, evaluators still running | 1+ |
| `provisional` | AI + embedding complete, awaiting human feedback | 2 (automated) |
| `final` | Human feedback submitted OR admin marks final | Score unchanged, feedback added |

### Explanation JSON Structure

```json
{
  "signals": [
    {
      "type": "ai",
      "evaluator_id": "gpt-4.1",
      "raw_score": 82,
      "normalized_score": 0.82,
      "weight": 0.42,
      "confidence": 0.85,
      "effective_weight": 0.357,
      "contribution_pct": "38.2%"
    },
    {
      "type": "embedding",
      "evaluator_id": "text-embedding-3-small",
      "raw_score": 71,
      "normalized_score": 0.71,
      "weight": 0.18,
      "confidence": 0.6,
      "effective_weight": 0.108,
      "contribution_pct": "8.2%"
    },
    {
      "type": "human",
      "evaluator_id": "mentor_user_42",
      "score": null,
      "feedback": "Good structure but needs deeper analysis of the counter-arguments. Consider revisiting paragraph 3.",
      "criteria_feedback": {
        "Argument Quality": "Strong thesis but weak supporting evidence",
        "Structure": "Well organized, clear flow"
      },
      "contributes_to_score": false
    }
  ],
  "method": "weighted_average_with_dynamic_trust",
  "total_effective_weight": 0.465,
  "computed_at": "2026-04-09T14:30:00Z"
}
```

---

## Dynamic Trust Calibration

Since human review is feedback-only (no scores), trust calibration is based on **inter-signal agreement** between AI and embedding evaluators:

```python
ai_signal = get_signal(evaluation_id, "ai")
embedding_signal = get_signal(evaluation_id, "embedding")

# Agreement between AI and embedding signals
agreement = 1 - abs(ai_signal.normalized_score - embedding_signal.normalized_score)

# When signals agree strongly (>0.85), both gain trust
# When signals disagree (<0.5), both lose trust slightly — signals admin to review weights
for evaluator_type in ["ai", "embedding"]:
    trust = get_trust(org_id, evaluator_type)
    trust.total_agreements += agreement
    trust.total_evaluations += 1
    
    base_weights = {"ai": 0.65, "embedding": 0.35}
    raw_weight = base_weights[evaluator_type] * (trust.total_agreements / trust.total_evaluations)
    trust.trust_weight = max(0.10, raw_weight)  # Floor: never drop below 10%
```

**Admin override**: Since human feedback is qualitative, admins can manually adjust AI vs embedding weights based on their observation of feedback quality via `PUT /evaluations/trust`.

**New orgs**: Start with default weights (ai=0.65, embedding=0.35). Trust evolves based on signal agreement patterns.

**Minimum floor**: Trust weights never drop below 0.10 to prevent one signal from dominating entirely.

---

## API Endpoints

### Evaluation Lifecycle

**`POST /evaluations/`** — Trigger multi-signal evaluation
```json
// Request
{ "user_id": 1, "task_id": 5, "question_id": 12, "submission_content": "..." }

// Response (streamed — AI feedback first, then provisional score)
{ "evaluation_id": 42, "status": "provisional", "provisional_score": 77.3, "ai_feedback": "..." }
```

**`GET /evaluations/{evaluation_id}`** — Get evaluation with all signals
```json
{
  "id": 42,
  "status": "final",
  "final_score": 78.5,
  "max_score": 100,
  "pass_score": 60,
  "passed": true,
  "signals": [...],
  "explanation": {...}
}
```

**`GET /evaluations/user/{user_id}/task/{task_id}`** — Get user's evaluation for a task

### Human Review

**`GET /evaluations/review/pending`** — List submissions awaiting mentor review
- Query params: `org_id`, `cohort_id` (optional), `limit`, `offset`
- Returns submissions with AI + embedding scores shown as advisory context

**`POST /evaluations/{evaluation_id}/review`** — Submit human feedback (no scores)
```json
// Request
{
  "reviewer_user_id": 42,
  "criteria_feedback": {
    "Argument Quality": "Strong thesis but weak supporting evidence in paragraph 2",
    "Structure": "Well organized, clear flow between sections"
  },
  "overall_feedback": "Good effort. Focus on strengthening evidence and counter-arguments."
}
```

### Trust Management

**`GET /evaluations/trust`** — Get current trust weights for org
**`PUT /evaluations/trust`** — Admin override of trust weights

### Explainability

**`GET /evaluations/{evaluation_id}/explain`** — Detailed breakdown for student
- Returns the explanation JSON with human-readable descriptions

---

## Integration with Existing System

### Minimal disruption to current flow:

1. **`POST /ai/chat` (quiz)** — Unchanged streaming behavior for students. Behind the scenes, after AI response completes, create evaluation record + AI signal + trigger embedding evaluator.

2. **`POST /ai/assignment`** — Same. After the three-phase assignment flow completes, create evaluation with AI signal and run embedding evaluator.

3. **New evaluation fields surfaced** — Frontend can optionally call `/evaluations/{id}` to show the multi-signal breakdown alongside existing feedback.

4. **Mentor dashboard** — New page calling `/evaluations/review/pending` to show submissions needing human review with AI/embedding scores as advisory.

### Files to modify:
- `src/api/db/__init__.py` — Add new table schemas
- `src/api/models.py` — Add Pydantic models for evaluation, signals, trust
- `src/api/routes/ai.py` — After AI response, create evaluation + trigger pipeline
- `src/api/db/task.py` — Add evaluation DB operations (or new `src/api/db/evaluation.py`)

### New files:
- `src/api/evaluators/__init__.py` — Evaluator base class
- `src/api/evaluators/ai_evaluator.py` — Refactored AI evaluation logic
- `src/api/evaluators/embedding_evaluator.py` — New embedding-based evaluator
- `src/api/evaluators/human_evaluator.py` — Human review signal handler
- `src/api/evaluators/aggregator.py` — Weighted aggregation + trust update
- `src/api/routes/evaluation.py` — New API endpoints
- `src/api/db/evaluation.py` — Database operations for evaluations
- `src/api/prompts/evaluation.py` — Any new prompt templates

---

## Verification Plan

1. **Unit tests**: Test aggregator with mock signals (equal weights, skewed weights, missing signals)
2. **Integration test**: Submit a text response → verify AI signal created → verify embedding signal created → verify provisional score
3. **Human review test**: Submit mentor review → verify score recalculation → verify trust weight update
4. **Edge cases**: 
   - Only AI signal (no reference answer for embedding, no human review yet)
   - Human score drastically different from AI → verify trust adjustment
   - New org with no trust history → verify defaults apply
5. **API tests**: All new endpoints return correct shapes and status codes

---

## Scope

**In scope (this spec)**:
- Text/essay evaluation with three signal types
- Progressive scoring (provisional → final)
- Weighted aggregation with dynamic trust
- Explainable score breakdown
- Human review workflow
- Integration with existing AI evaluation flow

**Out of scope (future)**:
- Code execution / automated test runner evaluator
- Peer review evaluator
- WebSocket notifications for score updates
- Historical trust analytics dashboard
- Batch re-evaluation when trust weights change
