import struct
from unittest.mock import patch, MagicMock
from api.engine.transcriber import transcribe_audio, _empty_result


def _make_wav_bytes(duration_seconds=5.0, sample_rate=16000):
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


def test_empty_audio_returns_empty_result():
    result = transcribe_audio(b"")
    assert result["transcript"] == ""
    assert result["word_count"] == 0


def test_too_short_audio_returns_empty_result():
    result = transcribe_audio(b"\x00" * 50)
    assert result["transcript"] == ""
    assert result["word_count"] == 0


def test_successful_transcription():
    wav = _make_wav_bytes(duration_seconds=5.0)

    mock_result = {
        "text": "Hello world this is a test",
        "segments": [
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0},
                    {"word": "this", "start": 1.0, "end": 1.3},
                    {"word": "is", "start": 1.3, "end": 1.5},
                    {"word": "a", "start": 1.5, "end": 1.6},
                    {"word": "test", "start": 1.6, "end": 2.0},
                ],
            }
        ],
    }

    mock_model = MagicMock()
    mock_model.transcribe.return_value = mock_result

    with patch("api.engine.transcriber._get_model", return_value=mock_model):
        result = transcribe_audio(wav)

    assert result["transcript"] == "Hello world this is a test"
    assert result["word_count"] == 6
    assert result["duration_seconds"] == 5.0
    assert len(result["words"]) == 6
    assert result["words"][0]["word"] == "Hello"


def test_transcription_failure_returns_empty_result():
    wav = _make_wav_bytes(duration_seconds=2.0)

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = Exception("Model error")

    with patch("api.engine.transcriber._get_model", return_value=mock_model):
        result = transcribe_audio(wav)

    assert result["transcript"] == ""
    assert result["word_count"] == 0
    assert "error" in result


def test_empty_result_helper():
    result = _empty_result(duration_seconds=3.5, error="test error")
    assert result["duration_seconds"] == 3.5
    assert result["error"] == "test error"
    assert result["transcript"] == ""
