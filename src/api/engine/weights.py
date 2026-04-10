TEXT_WEIGHTS = {"keyword": 0.15, "semantic": 0.25, "llm": 0.60}
TEXT_WEIGHTS_WITH_HUMAN = {"keyword": 0.10, "semantic": 0.15, "llm": 0.40, "human": 0.35}

AUDIO_WEIGHTS = {"fluency": 0.10, "confidence": 0.10, "semantic": 0.20, "llm": 0.60}
AUDIO_WEIGHTS_WITH_HUMAN = {"fluency": 0.08, "confidence": 0.07, "semantic": 0.15, "llm": 0.35, "human": 0.35}

CODE_WEIGHTS = {"autograder": 0.35, "static": 0.10, "llm": 0.55}
CODE_WEIGHTS_WITH_HUMAN = {"autograder": 0.25, "static": 0.05, "llm": 0.35, "human": 0.35}

_WEIGHT_MAP = {
    "text":  (TEXT_WEIGHTS, TEXT_WEIGHTS_WITH_HUMAN),
    "audio": (AUDIO_WEIGHTS, AUDIO_WEIGHTS_WITH_HUMAN),
    "code":  (CODE_WEIGHTS, CODE_WEIGHTS_WITH_HUMAN),
}

def get_weights_for_type(submission_type: str, has_human: bool = False) -> dict[str, float]:
    pair = _WEIGHT_MAP.get(submission_type, _WEIGHT_MAP["text"])
    return pair[1] if has_human else pair[0]
