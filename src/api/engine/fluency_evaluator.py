"""
Tier-1 Fluency Evaluator — analyzes audio delivery quality.
Runs locally with no external API calls.

Metrics: filler words, silence ratio, speaking pace (WPM), long pauses, duration.
"""

import io
from pydub import AudioSegment
from pydub.silence import detect_silence

# Common English filler words
FILLER_WORDS = {"um", "uh", "uh-huh", "hmm"}
# Filler phrases checked via bigram/trigram matching on transcript
FILLER_PHRASES = {"you know", "i mean", "sort of", "kind of"}
# Words that are only fillers when used as discourse markers (high frequency)
DISCOURSE_FILLERS = {"like", "so", "basically", "actually", "right", "okay"}
DISCOURSE_FILLER_THRESHOLD = 0.06  # flag only if > 6% of words

# Silence detection parameters
MIN_SILENCE_LEN_MS = 500
SILENCE_THRESH_DB = -40
LONG_PAUSE_MS = 3000


def evaluate_fluency(
    audio_data: bytes,
    transcript_result: dict,
    max_score: float,
    minimum_duration_seconds: float = 10.0,
) -> dict:
    """
    Evaluate fluency of an audio submission.

    Args:
        audio_data: Raw WAV bytes
        transcript_result: Output from transcriber.transcribe_audio()
        max_score: Maximum score for this evaluator
        minimum_duration_seconds: Minimum expected duration

    Returns dict with: score, max_score, normalized_score, confidence,
    filler_count, filler_ratio, silence_ratio, words_per_minute,
    long_pauses, duration_seconds, deductions, filler_words_found.
    """
    transcript = transcript_result.get("transcript", "")
    words_list = transcript_result.get("words", [])
    word_count = transcript_result.get("word_count", 0)
    duration = transcript_result.get("duration_seconds", 0.0)

    # Get duration from audio if not in transcript
    if duration <= 0:
        try:
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
            duration = len(audio_segment) / 1000.0
        except Exception:
            duration = 0.0

    # Edge case: audio too short
    if duration < 3.0:
        return _too_short_result(max_score, duration)

    # --- Filler word detection ---
    filler_count, filler_words_found = _count_fillers(transcript, word_count)
    filler_ratio = filler_count / word_count if word_count > 0 else 0.0

    # --- Silence analysis ---
    silence_ratio, long_pauses = _analyze_silence(audio_data, duration)

    # --- Speaking pace ---
    wpm = (word_count / (duration / 60.0)) if duration > 0 else 0.0

    # --- Scoring (spec deduction rules) ---
    deductions = {}
    total_deductions = 0.0

    # Filler penalty
    if filler_ratio > 0.15:
        deductions["filler"] = 3.0
        total_deductions += 3.0
    elif filler_ratio > 0.10:
        deductions["filler"] = 2.0
        total_deductions += 2.0
    elif filler_ratio > 0.05:
        deductions["filler"] = 1.0
        total_deductions += 1.0

    # Silence penalty
    if silence_ratio > 0.50:
        deductions["silence"] = 2.0
        total_deductions += 2.0
    elif silence_ratio > 0.30:
        deductions["silence"] = 1.0
        total_deductions += 1.0

    # Pace penalty
    if wpm < 80:
        deductions["pace_slow"] = 1.0
        total_deductions += 1.0
    elif wpm > 200:
        deductions["pace_fast"] = 1.0
        total_deductions += 1.0

    # Duration penalty
    if duration < minimum_duration_seconds:
        deductions["duration"] = 2.0
        total_deductions += 2.0

    score = max(0.0, max_score - total_deductions)
    normalized = score / max_score if max_score > 0 else 0.0

    return {
        "score": round(score, 2),
        "max_score": max_score,
        "normalized_score": round(normalized, 4),
        "confidence": 0.90,
        "filler_count": filler_count,
        "filler_ratio": round(filler_ratio, 4),
        "silence_ratio": round(silence_ratio, 4),
        "words_per_minute": round(wpm, 1),
        "long_pauses": long_pauses,
        "duration_seconds": round(duration, 2),
        "deductions": deductions,
        "filler_words_found": filler_words_found,
    }


def _count_fillers(transcript: str, word_count: int) -> tuple[int, list[str]]:
    """Count filler words and phrases in transcript."""
    if not transcript:
        return 0, []

    text_lower = transcript.lower()
    words = text_lower.split()
    found = []
    count = 0

    # Check single-word fillers
    for word in words:
        cleaned = word.strip(".,!?;:'\"")
        if cleaned in FILLER_WORDS:
            count += 1
            if cleaned not in found:
                found.append(cleaned)

    # Check discourse fillers (only if above threshold)
    for filler in DISCOURSE_FILLERS:
        filler_occurrences = sum(1 for w in words if w.strip(".,!?;:'\"") == filler)
        ratio = filler_occurrences / word_count if word_count > 0 else 0
        if ratio > DISCOURSE_FILLER_THRESHOLD:
            count += filler_occurrences
            if filler not in found:
                found.append(filler)

    # Check multi-word filler phrases
    for phrase in FILLER_PHRASES:
        phrase_count = text_lower.count(phrase)
        if phrase_count > 0:
            count += phrase_count
            if phrase not in found:
                found.append(phrase)

    return count, found


def _analyze_silence(audio_data: bytes, total_duration: float) -> tuple[float, int]:
    """Analyze silence in audio. Returns (silence_ratio, long_pause_count)."""
    if total_duration <= 0:
        return 0.0, 0

    try:
        audio = AudioSegment.from_wav(io.BytesIO(audio_data))
        silent_ranges = detect_silence(
            audio,
            min_silence_len=MIN_SILENCE_LEN_MS,
            silence_thresh=SILENCE_THRESH_DB,
        )

        total_silence_ms = sum(end - start for start, end in silent_ranges)
        silence_ratio = (total_silence_ms / 1000.0) / total_duration

        long_pauses = sum(1 for start, end in silent_ranges if (end - start) >= LONG_PAUSE_MS)

        return min(silence_ratio, 1.0), long_pauses

    except Exception:
        return 0.0, 0


def _too_short_result(max_score: float, duration: float) -> dict:
    return {
        "score": 0.0,
        "max_score": max_score,
        "normalized_score": 0.0,
        "confidence": 0.90,
        "filler_count": 0,
        "filler_ratio": 0.0,
        "silence_ratio": 0.0,
        "words_per_minute": 0.0,
        "long_pauses": 0,
        "duration_seconds": round(duration, 2),
        "deductions": {"too_short": max_score},
        "filler_words_found": [],
    }
