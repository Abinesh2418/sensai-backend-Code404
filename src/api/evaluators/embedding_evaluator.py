"""
Embedding Evaluator — scores submissions by semantic similarity to reference answers.

Uses OpenAI text-embedding-3-small to embed both the student response and
reference answer(s), then computes cosine similarity as the score.
"""

import math
from typing import Optional, Dict, List
from openai import AsyncOpenAI
from api.evaluators import BaseEvaluator, EvaluationSignal
from api.utils.logging import logger

EMBEDDING_MODEL = "text-embedding-3-small"


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors without numpy."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using OpenAI API."""
    client = AsyncOpenAI()

    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


class EmbeddingEvaluator(BaseEvaluator):
    """Scores submissions by semantic similarity to reference answers."""

    async def evaluate(
        self,
        submission_content: str,
        scorecard: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> Optional[EvaluationSignal]:
        """
        Evaluate by comparing submission embedding against reference answer embeddings.

        context should contain:
        - reference_answers: list of reference answer strings
        - max_score: maximum score for this evaluation
        """
        if not context:
            return None

        reference_answers = context.get("reference_answers", [])
        max_score = context.get("max_score", 100.0)

        if not reference_answers or not submission_content:
            return None

        # Filter out empty references
        reference_answers = [r for r in reference_answers if r and r.strip()]
        if not reference_answers:
            return None

        try:
            # Embed all texts in a single API call
            all_texts = [submission_content] + reference_answers
            embeddings = await get_embeddings(all_texts)

            submission_embedding = embeddings[0]
            reference_embeddings = embeddings[1:]

            # Compute similarity against each reference and take the best match
            similarities = [
                cosine_similarity(submission_embedding, ref_emb)
                for ref_emb in reference_embeddings
            ]

            best_similarity = max(similarities)

            # Clamp to [0, 1] range (cosine similarity can be negative for very dissimilar texts)
            best_similarity = max(0.0, min(1.0, best_similarity))

            score = best_similarity * max_score
            num_refs = len(reference_answers)

            # Confidence scales with number of reference answers
            confidence = min(0.8, 0.3 + 0.15 * num_refs)

            return EvaluationSignal(
                evaluator_type="embedding",
                evaluator_id=EMBEDDING_MODEL,
                score=score,
                max_score=max_score,
                normalized_score=best_similarity,
                confidence=confidence,
                feedback=f"Semantic similarity: {best_similarity:.2f} (best of {num_refs} reference{'s' if num_refs > 1 else ''})",
                metadata={
                    "model": EMBEDDING_MODEL,
                    "cosine_similarity": best_similarity,
                    "all_similarities": similarities,
                    "reference_count": num_refs,
                },
            )

        except Exception as e:
            logger.error(f"Embedding evaluation failed: {e}")
            return None
