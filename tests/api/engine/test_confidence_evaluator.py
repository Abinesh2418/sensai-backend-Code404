import struct
from api.engine.confidence_evaluator import (
    evaluate_confidence,
    _count_hedging,
    _count_false_starts,
)


def _make_wav_bytes(duration_seconds=10.0, sample_rate=16000):
    """Create a minimal WAV file with silence for testing."""
    num_samples = int(sample_rate * duration_seconds)
    data_size = num_samples * 2
    wav = b"RIFF"
    wav += struct.pack("<I", 36 + data_size)
    wav += b"WAVEfmt "
    wav += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    wav += b"data"
    wav += struct.pack("<I", data_size)
    wav += b"\x00\x00" * num_samples
    return wav


def _make_transcript(text, duration=10.0):
    words = text.split()
    return {
        "transcript": text,
        "words": [{"word": w, "start": i * 0.3, "end": (i + 1) * 0.3} for i, w in enumerate(words)],
        "duration_seconds": duration,
        "word_count": len(words),
    }


def test_too_short_audio():
    wav = _make_wav_bytes(duration_seconds=2.0)
    transcript = _make_transcript("hi", duration=2.0)
    result = evaluate_confidence(wav, transcript, max_score=10.0)
    assert result["score"] == 0.0
    assert "too_short" in result["deductions"]


def test_confident_delivery():
    wav = _make_wav_bytes(duration_seconds=30.0)
    text = "The answer is recursion which uses a base case to terminate and a recursive case to break down the problem into smaller subproblems each time"
    transcript = _make_transcript(text, duration=30.0)
    result = evaluate_confidence(wav, transcript, max_score=10.0)
    # No hedging, no false starts in this text
    assert result["hedging_phrases_count"] == 0
    assert result["false_starts"] == 0
    assert result["score"] >= 9.0  # should be high (may get +1 bonus)


def test_hedging_detection():
    text = "I think maybe the answer is probably recursion I guess and sort of like a base case maybe"
    transcript = _make_transcript(text, duration=15.0)
    wav = _make_wav_bytes(duration_seconds=15.0)
    result = evaluate_confidence(wav, transcript, max_score=10.0)
    assert result["hedging_phrases_count"] > 0
    assert result["hedging_ratio"] > 0


def test_heavy_hedging_penalty():
    # 50%+ hedging words
    text = "I think maybe probably I guess sort of kind of maybe I think possibly probably I guess maybe sort of"
    transcript = _make_transcript(text, duration=15.0)
    wav = _make_wav_bytes(duration_seconds=15.0)
    result = evaluate_confidence(wav, transcript, max_score=10.0)
    assert result["hedging_ratio"] > 0.20
    assert "hedging" in result["deductions"]


def test_false_starts_detection():
    text = "I mean the the answer is well actually no wait it is recursion I mean the base case"
    transcript = _make_transcript(text, duration=15.0)
    wav = _make_wav_bytes(duration_seconds=15.0)
    result = evaluate_confidence(wav, transcript, max_score=10.0)
    assert result["false_starts"] > 0


def test_count_hedging_basic():
    count, total_words = _count_hedging("I think maybe the answer is probably correct I guess")
    assert count >= 2  # "I think maybe", "probably", "I guess"
    assert total_words > 0


def test_count_hedging_empty():
    count, total_words = _count_hedging("")
    assert count == 0
    assert total_words == 0


def test_count_false_starts_repeated_words():
    count = _count_false_starts("the the answer is is correct")
    assert count >= 2  # "the the" and "is is"


def test_count_false_starts_patterns():
    count = _count_false_starts("I mean the answer is well actually recursion")
    assert count >= 2  # "I mean" and "well actually"


def test_strong_delivery_bonus():
    wav = _make_wav_bytes(duration_seconds=30.0)
    text = "Recursion is a technique where a function calls itself with a base case to stop and a recursive case to continue processing"
    transcript = _make_transcript(text, duration=30.0)
    result = evaluate_confidence(wav, transcript, max_score=10.0)
    # Clean delivery → bonus possible (depends on volume analysis with silent wav)
    assert result["score"] <= result["max_score"]


def test_score_never_below_zero():
    wav = _make_wav_bytes(duration_seconds=10.0)
    text = "I think maybe I guess probably sort of I mean no wait I think maybe I guess well actually no no sorry"
    transcript = _make_transcript(text, duration=10.0)
    result = evaluate_confidence(wav, transcript, max_score=10.0)
    assert result["score"] >= 0.0


def test_output_shape():
    wav = _make_wav_bytes(duration_seconds=15.0)
    transcript = _make_transcript("this is a test answer for confidence", duration=15.0)
    result = evaluate_confidence(wav, transcript, max_score=10.0)
    assert "score" in result
    assert "max_score" in result
    assert "normalized_score" in result
    assert "confidence" in result
    assert "hedging_phrases_count" in result
    assert "hedging_ratio" in result
    assert "volume_drop_count" in result
    assert "false_starts" in result
    assert "false_starts_per_minute" in result
    assert "voice_stability" in result
    assert "strong_delivery_bonus" in result
    assert "deductions" in result
