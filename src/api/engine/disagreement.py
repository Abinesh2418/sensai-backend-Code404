from typing import Optional

def detect_disagreements(scores: dict[str, Optional[float]]) -> list[dict]:
    active = {k: v for k, v in scores.items() if v is not None}
    keys = sorted(active.keys())
    disagreements = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key, b_key = keys[i], keys[j]
            diff = abs(active[a_key] - active[b_key])
            if diff <= 0.15:
                continue
            severity = "minor" if diff <= 0.30 else "major"
            resolution = "proceed_weighted_average" if severity == "minor" else "queue_human_review"
            disagreements.append({
                "evaluator_a": a_key, "evaluator_b": b_key,
                "score_a": round(active[a_key], 4), "score_b": round(active[b_key], 4),
                "diff": round(diff, 4), "severity": severity, "resolution": resolution,
            })
    return disagreements
