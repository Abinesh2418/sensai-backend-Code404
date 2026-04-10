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
