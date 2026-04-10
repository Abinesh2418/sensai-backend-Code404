"""
Tier-1 Confidence Evaluator — analyzes speaker confidence from audio.
Runs locally with no external API calls.

Metrics: hedging language, volume drops, false starts, voice stability.
"""

import io
import math
from pydub import AudioSegment

# Hedging phrases that indicate uncertainty
HEDGING_PHRASES = [
    "i think maybe",
    "i'm not sure but",
    "i'm not sure",
    "it could be",
    "i guess",
    "kind of",
    "sort of",
    "i think",
    "maybe",
    "possibly",
    "probably",
    "perhaps",
    "not really sure",
    "i don't know",
    "i believe",
]

# False start indicators in transcript
FALSE_START_PATTERNS = [
    "i mean",
    "well actually",
    "no wait",
    "sorry",
    "let me rephrase",
    "what i meant",
    "no no",
]

# Audio analysis parameters
CHUNK_MS = 500  # Analyze audio in 500ms chunks
VOLUME_DROP_THRESHOLD = 0.50  # >50% RMS drop = significant


def evaluate_confidence(
    audio_data: bytes,
    transcript_result: dict,
    max_score: float,
) -> dict:
    """
    Evaluate speaker confidence from audio submission.

    Args:
        audio_data: Raw WAV bytes
        transcript_result: Output from transcriber.transcribe_audio()
        max_score: Maximum score for this evaluator

    Returns dict with: score, max_score, normalized_score, confidence,
    hedging_phrases_count, hedging_ratio, volume_drop_count,
    false_starts, false_starts_per_minute, voice_stability,
    strong_delivery_bonus, deductions.
    """
    transcript = transcript_result.get("transcript", "")
    duration = transcript_result.get("duration_seconds", 0.0)
    word_count = transcript_result.get("word_count", 0)

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

    # --- Hedging language detection ---
    hedging_count, hedging_words_total = _count_hedging(transcript)
    hedging_ratio = hedging_words_total / word_count if word_count > 0 else 0.0

    # --- False start detection ---
    false_starts = _count_false_starts(transcript)
    duration_minutes = duration / 60.0
    false_starts_per_minute = false_starts / duration_minutes if duration_minutes > 0 else 0.0

    # --- Volume analysis ---
    volume_drop_count, voice_stability = _analyze_volume(audio_data)

    # --- Scoring (spec deduction rules) ---
    deductions = {}
    total_deductions = 0.0

    # Hedging penalty
    if hedging_ratio > 0.40:
        deductions["hedging"] = 2.0
        total_deductions += 2.0
    elif hedging_ratio > 0.20:
        deductions["hedging"] = 1.0
        total_deductions += 1.0

    # Volume drop penalty
    if volume_drop_count > 0:
        deductions["volume_drops"] = 1.0
        total_deductions += 1.0

    # False starts penalty
    if false_starts_per_minute > 6:
        deductions["false_starts"] = 2.0
        total_deductions += 2.0
    elif false_starts_per_minute > 3:
        deductions["false_starts"] = 1.0
        total_deductions += 1.0

    # Strong delivery bonus
    strong_delivery = (
        hedging_ratio < 0.05
        and false_starts_per_minute < 1.0
        and voice_stability > 0.80
        and volume_drop_count == 0
    )
    bonus = 1.0 if strong_delivery else 0.0

    score = max(0.0, min(max_score, max_score - total_deductions + bonus))
    normalized = score / max_score if max_score > 0 else 0.0

    return {
        "score": round(score, 2),
        "max_score": max_score,
        "normalized_score": round(normalized, 4),
        "confidence": 0.90,
        "hedging_phrases_count": hedging_count,
        "hedging_ratio": round(hedging_ratio, 4),
        "volume_drop_count": volume_drop_count,
        "false_starts": false_starts,
        "false_starts_per_minute": round(false_starts_per_minute, 2),
        "voice_stability": round(voice_stability, 4),
        "strong_delivery_bonus": strong_delivery,
        "deductions": deductions,
    }


def _count_hedging(transcript: str) -> tuple[int, int]:
    """
    Count hedging phrases in transcript.
    Returns (phrase_count, total_hedging_words) where total_hedging_words
    is the sum of words in all matched hedging phrases.
    """
    if not transcript:
        return 0, 0

    text_lower = transcript.lower()
    phrase_count = 0
    total_words = 0

    # Sort phrases longest-first to match longer phrases before shorter ones
    for phrase in sorted(HEDGING_PHRASES, key=len, reverse=True):
        count = text_lower.count(phrase)
        if count > 0:
            phrase_count += count
            total_words += count * len(phrase.split())
            # Remove matched phrases to avoid double-counting
            text_lower = text_lower.replace(phrase, " " * len(phrase))

    return phrase_count, total_words


def _count_false_starts(transcript: str) -> int:
    """Count false start patterns in transcript."""
    if not transcript:
        return 0

    text_lower = transcript.lower()
    count = 0

    for pattern in FALSE_START_PATTERNS:
        count += text_lower.count(pattern)

    # Also detect repeated words (stuttering): "the the", "I I"
    words = text_lower.split()
    for i in range(len(words) - 1):
        w1 = words[i].strip(".,!?;:'\"")
        w2 = words[i + 1].strip(".,!?;:'\"")
        if w1 and w1 == w2 and len(w1) > 1:
            count += 1

    return count


def _analyze_volume(audio_data: bytes) -> tuple[int, float]:
    """
    Analyze volume patterns in audio.
    Returns (significant_drop_count, voice_stability 0.0-1.0).
    """
    try:
        audio = AudioSegment.from_wav(io.BytesIO(audio_data))
    except Exception:
        return 0, 0.5

    duration_ms = len(audio)
    if duration_ms < CHUNK_MS * 2:
        return 0, 0.5

    # Compute RMS per chunk
    rms_values = []
    for start_ms in range(0, duration_ms - CHUNK_MS + 1, CHUNK_MS):
        chunk = audio[start_ms:start_ms + CHUNK_MS]
        rms = chunk.rms
        rms_values.append(rms)

    if not rms_values:
        return 0, 0.5

    # Filter out silent chunks (very low RMS) from stability calculation
    non_silent = [r for r in rms_values if r > 50]
    if len(non_silent) < 2:
        return 0, 0.5

    # Volume drops: count consecutive chunk pairs with >50% RMS decrease
    drop_count = 0
    for i in range(len(non_silent) - 1):
        if non_silent[i] > 0:
            drop_ratio = 1.0 - (non_silent[i + 1] / non_silent[i])
            if drop_ratio > VOLUME_DROP_THRESHOLD:
                drop_count += 1

    # Voice stability: 1 - coefficient of variation of RMS
    mean_rms = sum(non_silent) / len(non_silent)
    if mean_rms > 0:
        variance = sum((r - mean_rms) ** 2 for r in non_silent) / len(non_silent)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_rms
        # Map CV to stability: CV=0 → 1.0, CV=1 → 0.0
        stability = max(0.0, min(1.0, 1.0 - cv))
    else:
        stability = 0.5

    return drop_count, stability


def _too_short_result(max_score: float, duration: float) -> dict:
    return {
        "score": 0.0,
        "max_score": max_score,
        "normalized_score": 0.0,
        "confidence": 0.90,
        "hedging_phrases_count": 0,
        "hedging_ratio": 0.0,
        "volume_drop_count": 0,
        "false_starts": 0,
        "false_starts_per_minute": 0.0,
        "voice_stability": 0.0,
        "strong_delivery_bonus": False,
        "deductions": {"too_short": max_score},
    }
