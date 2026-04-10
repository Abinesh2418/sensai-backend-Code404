from api.engine.weights import TEXT_WEIGHTS, TEXT_WEIGHTS_WITH_HUMAN, get_weights_for_type

def test_text_weights_without_human_sum_to_1():
    total = sum(TEXT_WEIGHTS.values())
    assert abs(total - 1.0) < 0.001

def test_text_weights_without_human_values():
    assert TEXT_WEIGHTS == {"keyword": 0.15, "semantic": 0.25, "llm": 0.60}

def test_text_weights_with_human_sum_to_1():
    total = sum(TEXT_WEIGHTS_WITH_HUMAN.values())
    assert abs(total - 1.0) < 0.001

def test_text_weights_with_human_values():
    assert TEXT_WEIGHTS_WITH_HUMAN == {"keyword": 0.10, "semantic": 0.15, "llm": 0.40, "human": 0.35}

def test_get_weights_for_text_no_human():
    w = get_weights_for_type("text", has_human=False)
    assert w == TEXT_WEIGHTS

def test_get_weights_for_text_with_human():
    w = get_weights_for_type("text", has_human=True)
    assert w == TEXT_WEIGHTS_WITH_HUMAN
