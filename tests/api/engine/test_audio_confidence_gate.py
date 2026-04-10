from api.engine.confidence_gate import (
    compute_audio_tier1_confidence,
    should_run_llm_audio,
    should_queue_human_review_audio,
)


def test_all_signals_agree_high():
    conf = compute_audio_tier1_confidence(0.90, 0.90, 0.90)
    # agreement = 1 - 0 = 1.0, extremity = |0.90 - 0.5| * 2 = 0.80
    # confidence = 1.0 * 0.6 + 0.80 * 0.4 = 0.92
    assert abs(conf - 0.92) < 0.01


def test_all_signals_agree_low():
    conf = compute_audio_tier1_confidence(0.10, 0.10, 0.10)
    # agreement = 1.0, extremity = |0.10 - 0.5| * 2 = 0.80
    # confidence = 1.0 * 0.6 + 0.80 * 0.4 = 0.92
    assert abs(conf - 0.92) < 0.01


def test_signals_disagree():
    conf = compute_audio_tier1_confidence(0.90, 0.40, 0.70)
    # max pairwise diff = |0.90 - 0.40| = 0.50
    # agreement = 1 - 0.50 = 0.50
    # avg = (0.90 + 0.40 + 0.70) / 3 = 0.6667
    # extremity = |0.6667 - 0.5| * 2 = 0.3333
    # confidence = 0.50 * 0.6 + 0.3333 * 0.4 = 0.4333
    assert abs(conf - 0.4333) < 0.01


def test_single_signal():
    conf = compute_audio_tier1_confidence(0.80, None, None)
    # Only 1 signal: agreement = 1.0, extremity = |0.80 - 0.5| * 2 = 0.60
    # confidence = 1.0 * 0.6 + 0.60 * 0.4 = 0.84
    assert abs(conf - 0.84) < 0.01


def test_no_signals():
    conf = compute_audio_tier1_confidence(None, None, None)
    assert conf == 0.0


def test_two_signals():
    conf = compute_audio_tier1_confidence(0.85, 0.85, None)
    # agreement = 1.0, extremity = |0.85 - 0.5| * 2 = 0.70
    # confidence = 1.0 * 0.6 + 0.70 * 0.4 = 0.88
    assert abs(conf - 0.88) < 0.01


def test_skip_llm_when_confident():
    # All signals agree at 0.90 → confidence = 0.92 → skip LLM
    assert should_run_llm_audio(0.90, 0.90, 0.90) is False


def test_run_llm_when_uncertain():
    assert should_run_llm_audio(0.50, 0.60, 0.40) is True


def test_run_llm_when_any_signal_missing():
    assert should_run_llm_audio(0.90, 0.90, None) is True
    assert should_run_llm_audio(None, 0.90, 0.90) is True
    assert should_run_llm_audio(0.90, None, 0.90) is True


def test_queue_human_review_when_very_low():
    # All signals very low → confidence should be low
    # 0.20, 0.20, 0.20: agreement=1.0, extremity=|0.20-0.5|*2=0.60
    # confidence = 0.60*0.6 + 0.60*0.4 = 0.84 — NOT low enough
    # Need more disagreement:
    # 0.10, 0.50, 0.20: max_diff=0.40, agreement=0.60
    # avg=0.2667, extremity=|0.2667-0.5|*2=0.4667
    # confidence = 0.60*0.6 + 0.4667*0.4 = 0.5467 — still not <=0.40
    # 0.10, 0.60, 0.10: max_diff=0.50, agreement=0.50
    # avg=0.2667, extremity=0.4667
    # confidence = 0.50*0.6 + 0.4667*0.4 = 0.4867 — still not
    # 0.30, 0.80, 0.30: max_diff=0.50, agreement=0.50
    # avg=0.4667, extremity=|0.4667-0.5|*2=0.0667
    # confidence = 0.50*0.6 + 0.0667*0.4 = 0.3267 → YES <= 0.40
    assert should_queue_human_review_audio(0.30, 0.80, 0.30) is True


def test_no_human_review_when_confident():
    assert should_queue_human_review_audio(0.90, 0.90, 0.90) is False
