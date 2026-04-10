import io
import struct
from pydub import AudioSegment
from api.engine.fluency_evaluator import evaluate_fluency, _count_fillers


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
    word_entries = []
    time_per_word = duration / len(words) if words else 0
    for i, w in enumerate(words):
        word_entries.append({"word": w, "start": i * time_per_word, "end": (i + 1) * time_per_word})
    return {
        "transcript": text,
        "words": word_entries,
        "duration_seconds": duration,
        "word_count": len(words),
    }


def test_too_short_audio():
    wav = _make_wav_bytes(duration_seconds=2.0)
    transcript = _make_transcript("hi", duration=2.0)
    result = evaluate_fluency(wav, transcript, max_score=10.0)
    assert result["score"] == 0.0
    assert result["normalized_score"] == 0.0
    assert "too_short" in result["deductions"]


def test_perfect_fluency():
    # Note: a silent WAV will trigger silence deductions. The fluency evaluator
    # correctly detects silence in the audio even though the transcript has words.
    # For a "perfect" test we check: no fillers, WPM in range, duration above min.
    wav = _make_wav_bytes(duration_seconds=30.0)
    text = " ".join(["word"] * 60)  # 60 words in 30s = 120 WPM (ideal range)
    transcript = _make_transcript(text, duration=30.0)
    result = evaluate_fluency(wav, transcript, max_score=10.0, minimum_duration_seconds=10.0)
    # No fillers, WPM in range, duration above min
    assert result["filler_count"] == 0
    assert 80 <= result["words_per_minute"] <= 200
    assert "filler" not in result["deductions"]
    assert "pace_slow" not in result["deductions"]
    assert "pace_fast" not in result["deductions"]
    assert "duration" not in result["deductions"]
    assert result["confidence"] == 0.90
    # Score may have silence deduction since test WAV is all zeros (silence)
    assert result["score"] >= 8.0


def test_filler_words_detected():
    text = "um the answer is uh recursion and um like the base case"
    # 10 words total, 3 fillers (um, uh, um) = 30% filler ratio → -3
    transcript = _make_transcript(text, duration=15.0)
    wav = _make_wav_bytes(duration_seconds=15.0)
    result = evaluate_fluency(wav, transcript, max_score=10.0)
    assert result["filler_count"] > 0
    assert "um" in result["filler_words_found"]


def test_slow_pace_penalty():
    wav = _make_wav_bytes(duration_seconds=60.0)
    text = " ".join(["word"] * 50)  # 50 words in 60s = 50 WPM (too slow)
    transcript = _make_transcript(text, duration=60.0)
    result = evaluate_fluency(wav, transcript, max_score=10.0, minimum_duration_seconds=10.0)
    assert result["words_per_minute"] < 80
    assert "pace_slow" in result["deductions"]


def test_fast_pace_penalty():
    wav = _make_wav_bytes(duration_seconds=10.0)
    text = " ".join(["word"] * 50)  # 50 words in 10s = 300 WPM (too fast)
    transcript = _make_transcript(text, duration=10.0)
    result = evaluate_fluency(wav, transcript, max_score=10.0, minimum_duration_seconds=5.0)
    assert result["words_per_minute"] > 200
    assert "pace_fast" in result["deductions"]


def test_duration_penalty():
    wav = _make_wav_bytes(duration_seconds=5.0)
    text = " ".join(["word"] * 15)  # 15 words in 5s = 180 WPM
    transcript = _make_transcript(text, duration=5.0)
    result = evaluate_fluency(wav, transcript, max_score=10.0, minimum_duration_seconds=10.0)
    assert "duration" in result["deductions"]
    assert result["deductions"]["duration"] == 2.0


def test_count_fillers_basic():
    count, found = _count_fillers("um the answer is uh well you know basically", word_count=8)
    assert count >= 2  # at least um and uh
    assert "um" in found
    assert "uh" in found


def test_count_fillers_empty():
    count, found = _count_fillers("", word_count=0)
    assert count == 0
    assert found == []


def test_score_never_below_zero():
    wav = _make_wav_bytes(duration_seconds=4.0)
    # Lots of fillers + short + slow → many deductions
    text = "um uh um uh um uh um uh um uh"
    transcript = _make_transcript(text, duration=4.0)
    result = evaluate_fluency(wav, transcript, max_score=10.0, minimum_duration_seconds=60.0)
    assert result["score"] >= 0.0


def test_output_shape():
    wav = _make_wav_bytes(duration_seconds=15.0)
    transcript = _make_transcript("this is a test answer for fluency", duration=15.0)
    result = evaluate_fluency(wav, transcript, max_score=10.0)
    assert "score" in result
    assert "max_score" in result
    assert "normalized_score" in result
    assert "confidence" in result
    assert "filler_count" in result
    assert "filler_ratio" in result
    assert "silence_ratio" in result
    assert "words_per_minute" in result
    assert "long_pauses" in result
    assert "duration_seconds" in result
    assert "deductions" in result
    assert "filler_words_found" in result
