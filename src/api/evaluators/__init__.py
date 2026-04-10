from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class EvaluationSignal:
    """Common signal produced by all evaluators."""
    evaluator_type: str           # "ai", "embedding", "human"
    evaluator_id: Optional[str] = None   # model name or user ID
    score: Optional[float] = None        # raw score (None for human feedback-only)
    max_score: Optional[float] = None    # max possible on evaluator's scale
    normalized_score: Optional[float] = None  # 0.0-1.0
    confidence: Optional[float] = None   # 0.0-1.0
    criteria_scores: Optional[Dict] = None   # per-criterion breakdown
    feedback: Optional[str] = None       # textual feedback
    metadata: Optional[Dict] = None      # extra info (model version, etc.)


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    async def evaluate(
        self,
        submission_content: str,
        scorecard: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> Optional[EvaluationSignal]:
        """
        Evaluate a submission and return a signal.
        Returns None if evaluation cannot be performed (e.g., no reference answer).
        """
        ...
