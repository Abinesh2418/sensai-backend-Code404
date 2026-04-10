"""
Audio Transcription Module — uses local OpenAI Whisper model to transcribe audio
with word-level timestamps. Runs entirely locally, no API calls, no cost.

Shared by fluency, confidence, and semantic evaluators.
"""

import io
import tempfile
import os
import whisper
from pydub import AudioSegment
from api.utils.logging import logger

# Load the model once at module level (cached in memory after first load).
# "base" is a good balance of speed vs accuracy (~150MB).
# Options: "tiny" (~40MB), "base" (~150MB), "small" (~500MB), "medium" (~1.5GB)
_model = None


def _get_model(model_name: str = "base"):
    global _model
    if _model is None:
        logger.info(f"Loading local Whisper model: {model_name}")
        _model = whisper.load_model(model_name)
        logger.info("Whisper model loaded")
    return _model


def transcribe_audio(audio_data: bytes, language: str = "en") -> dict:
    """
    Transcribe audio using local Whisper model with word-level timestamps.
    Runs locally — no API calls, no cost.

    Returns:
        {
            "transcript": str,
            "words": [{"word": str, "start": float, "end": float}, ...],
            "duration_seconds": float,
            "word_count": int,
        }
    On failure returns empty result with transcript="" and word_count=0.
    """
    if not audio_data or len(audio_data) < 100:
        return _empty_result()

    # Get duration from audio data
    try:
        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
        duration_seconds = len(audio_segment) / 1000.0
    except Exception:
        duration_seconds = 0.0

    if duration_seconds < 0.5:
        return _empty_result(duration_seconds=duration_seconds)

    # Write to temp file (whisper.load_audio needs a file path)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        model = _get_model()
        result = model.transcribe(
            tmp_path,
            language=language,
            word_timestamps=True,
        )

        # Extract transcript
        transcript = (result.get("text") or "").strip()

        # Extract word-level timestamps from segments
        words = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": round(float(w.get("start", 0)), 3),
                    "end": round(float(w.get("end", 0)), 3),
                })

        word_count = len(words) if words else len(transcript.split())

        return {
            "transcript": transcript,
            "words": words,
            "duration_seconds": round(duration_seconds, 2),
            "word_count": word_count,
        }

    except Exception as e:
        logger.error(f"Audio transcription failed: {e}", exc_info=True)
        return _empty_result(duration_seconds=duration_seconds, error=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _empty_result(duration_seconds: float = 0.0, error: str = None) -> dict:
    result = {
        "transcript": "",
        "words": [],
        "duration_seconds": round(duration_seconds, 2),
        "word_count": 0,
    }
    if error:
        result["error"] = error
    return result
