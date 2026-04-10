# Weights mirror Tier-1 proportional share: keyword 15/(15+25), semantic 25/(15+25)
KEYWORD_WEIGHT = 0.375
SEMANTIC_WEIGHT = 0.625

# If weighted Tier-1 score >= this threshold, Tier-1 alone is sufficient → skip LLM
LLM_SKIP_THRESHOLD = 0.85

# If weighted Tier-1 score <= this threshold, flag for human review
HUMAN_REVIEW_THRESHOLD = 0.40


def compute_tier1_weighted_score(keyword_normalized, semantic_normalized) -> float:
    """Weighted combination of keyword and semantic scores (proportional to their eval weights)."""
    if keyword_normalized is None and semantic_normalized is None:
        return 0.0
    if keyword_normalized is None:
        return semantic_normalized
    if semantic_normalized is None:
        return keyword_normalized
    return keyword_normalized * KEYWORD_WEIGHT + semantic_normalized * SEMANTIC_WEIGHT


def should_run_llm(keyword_normalized, semantic_normalized) -> bool:
    """
    Run LLM when Tier-1 weighted score is below the skip threshold.
    Skip LLM only when both keyword and semantic confidently agree the answer is strong.
    """
    if keyword_normalized is None or semantic_normalized is None:
        return True  # incomplete Tier-1 → always use LLM
    tier1_score = compute_tier1_weighted_score(keyword_normalized, semantic_normalized)
    return tier1_score < LLM_SKIP_THRESHOLD


def should_queue_human_review(keyword_normalized, semantic_normalized) -> bool:
    """Flag for human review when Tier-1 weighted score is very low."""
    tier1_score = compute_tier1_weighted_score(keyword_normalized, semantic_normalized)
    return tier1_score <= HUMAN_REVIEW_THRESHOLD


# ---------------------------------------------------------------------------
# Audio Tier-1 Confidence Gate
# ---------------------------------------------------------------------------
# Audio has 3 Tier-1 signals: fluency, confidence, semantic.
# Uses spec formula: confidence = (agreement * 0.6) + (extremity * 0.4)

def compute_audio_tier1_confidence(
    fluency_normalized, confidence_normalized, semantic_normalized
) -> float:
    """
    Compute Tier-1 confidence for audio using the spec formula:
      agreement = 1 - max(pairwise diffs among available signals)
      extremity = |mean(signals) - 0.5| * 2
      confidence = (agreement * 0.6) + (extremity * 0.4)
    """
    scores = [s for s in [fluency_normalized, confidence_normalized, semantic_normalized] if s is not None]
    if not scores:
        return 0.0
    if len(scores) == 1:
        extremity = abs(scores[0] - 0.5) * 2
        return (1.0 * 0.6) + (extremity * 0.4)

    # Agreement: 1 - max pairwise difference
    max_diff = 0.0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            diff = abs(scores[i] - scores[j])
            if diff > max_diff:
                max_diff = diff
    agreement = 1.0 - max_diff

    # Extremity: how far the average is from 0.5 (0 at 50%, 1 at 0% or 100%)
    avg = sum(scores) / len(scores)
    extremity = abs(avg - 0.5) * 2

    return (agreement * 0.6) + (extremity * 0.4)


def should_run_llm_audio(fluency_normalized, confidence_normalized, semantic_normalized) -> bool:
    """Run LLM when audio Tier-1 confidence is below threshold or any signal is missing."""
    if fluency_normalized is None or confidence_normalized is None or semantic_normalized is None:
        return True
    confidence = compute_audio_tier1_confidence(fluency_normalized, confidence_normalized, semantic_normalized)
    return confidence < LLM_SKIP_THRESHOLD


def should_queue_human_review_audio(fluency_normalized, confidence_normalized, semantic_normalized) -> bool:
    """Flag for human review when audio Tier-1 confidence is very low."""
    confidence = compute_audio_tier1_confidence(fluency_normalized, confidence_normalized, semantic_normalized)
    return confidence <= HUMAN_REVIEW_THRESHOLD
