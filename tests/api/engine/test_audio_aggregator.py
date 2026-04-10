from api.engine.aggregator import aggregate_audio_signals


def test_audio_aggregation_without_human():
    result = aggregate_audio_signals(
        fluency_normalized=0.80, confidence_normalized=0.70,
        semantic_normalized=0.85, llm_normalized=0.90,
        human_normalized=None, max_score=30.0,
    )
    # Weights: fluency=0.10, confidence=0.10, semantic=0.20, llm=0.60
    expected = (0.80 * 0.10 + 0.70 * 0.10 + 0.85 * 0.20 + 0.90 * 0.60) * 30.0
    assert abs(result["final_score"] - expected) < 0.01
    assert result["weights_used"]["fluency"] == 0.10
    assert result["weights_used"]["confidence"] == 0.10
    assert result["weights_used"]["semantic"] == 0.20
    assert result["weights_used"]["llm"] == 0.60


def test_audio_aggregation_with_human():
    result = aggregate_audio_signals(
        fluency_normalized=0.80, confidence_normalized=0.70,
        semantic_normalized=0.85, llm_normalized=0.90,
        human_normalized=0.60, max_score=30.0,
    )
    # Weights: fluency=0.08, confidence=0.07, semantic=0.15, llm=0.35, human=0.35
    expected = (0.80 * 0.08 + 0.70 * 0.07 + 0.85 * 0.15 + 0.90 * 0.35 + 0.60 * 0.35) * 30.0
    assert abs(result["final_score"] - expected) < 0.01
    assert result["weights_used"]["human"] == 0.35
    assert result["has_human_review"] is True


def test_audio_llm_skipped_redistributes():
    result = aggregate_audio_signals(
        fluency_normalized=0.80, confidence_normalized=0.70,
        semantic_normalized=0.85, llm_normalized=None,
        human_normalized=None, max_score=10.0,
    )
    # Active: fluency=0.10, confidence=0.10, semantic=0.20 → sum=0.40
    # Redistributed: fluency=0.25, confidence=0.25, semantic=0.50
    assert abs(result["weights_used"]["fluency"] - 0.25) < 0.01
    assert abs(result["weights_used"]["confidence"] - 0.25) < 0.01
    assert abs(result["weights_used"]["semantic"] - 0.50) < 0.01
    expected = (0.80 * 0.25 + 0.70 * 0.25 + 0.85 * 0.50) * 10.0
    assert abs(result["final_score"] - expected) < 0.01


def test_audio_only_llm_active():
    result = aggregate_audio_signals(
        fluency_normalized=None, confidence_normalized=None,
        semantic_normalized=None, llm_normalized=0.75,
        human_normalized=None, max_score=10.0,
    )
    assert result["weights_used"]["llm"] == 1.0
    assert abs(result["final_score"] - 7.5) < 0.01


def test_audio_no_signals():
    result = aggregate_audio_signals(
        fluency_normalized=None, confidence_normalized=None,
        semantic_normalized=None, llm_normalized=None,
        human_normalized=None, max_score=10.0,
    )
    assert result["final_score"] == 0.0
    assert result["signals"] == []


def test_audio_aggregation_signal_details():
    result = aggregate_audio_signals(
        fluency_normalized=0.80, confidence_normalized=0.70,
        semantic_normalized=0.85, llm_normalized=0.90,
        human_normalized=None, max_score=30.0,
    )
    assert len(result["signals"]) == 4
    assert result["method"] == "spec_weighted_average"
    signal_names = [s["name"] for s in result["signals"]]
    assert "fluency" in signal_names
    assert "confidence" in signal_names
    assert "semantic" in signal_names
    assert "llm" in signal_names
