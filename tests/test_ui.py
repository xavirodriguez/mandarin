"""Unit tests for Gradio UI helper functions in src/interfaces/ui/gradio_app.py."""

import base64
import pytest
import numpy as np

from src.interfaces.ui.gradio_app import (
    preprocess_audio,
    parse_pinyin_and_tones,
    format_assessment_output,
    assess_pronunciation_ui,
)


def test_preprocess_audio_valid_mono():
    # 1 second of 44100 Hz mono sine wave
    sr = 44100
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    data = np.sin(2 * np.pi * 440 * t)

    b64_str = preprocess_audio((sr, data))
    assert isinstance(b64_str, str)
    raw_bytes = base64.b64decode(b64_str)

    # 16000 Hz * 1 sec * 2 bytes/sample (int16) = 32000 bytes
    pcm_array = np.frombuffer(raw_bytes, dtype=np.int16)
    assert len(pcm_array) == 16000


def test_preprocess_audio_stereo():
    # 1 second of 32000 Hz stereo audio
    sr = 32000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    ch1 = np.sin(2 * np.pi * 440 * t)
    ch2 = np.sin(2 * np.pi * 880 * t)
    stereo_data = np.column_stack([ch1, ch2])

    b64_str = preprocess_audio((sr, stereo_data))
    raw_bytes = base64.b64decode(b64_str)
    pcm_array = np.frombuffer(raw_bytes, dtype=np.int16)
    assert len(pcm_array) == 16000


def test_preprocess_audio_none_or_empty():
    with pytest.raises(ValueError, match="No se proporcionó ningún archivo"):
        preprocess_audio(None)

    with pytest.raises(ValueError, match="está vacío"):
        preprocess_audio((16000, np.array([])))


def test_parse_pinyin_and_tones_valid():
    pinyin, tones = parse_pinyin_and_tones("ní hǎo", "3 3")
    assert pinyin == ["ní", "hǎo"]
    assert tones == [3, 3]

    pinyin, tones = parse_pinyin_and_tones("xiè, xie", "4, 0")
    assert pinyin == ["xiè", "xie"]
    assert tones == [4, 0]


def test_parse_pinyin_and_tones_mismatch():
    with pytest.raises(ValueError, match="no coincide"):
        parse_pinyin_and_tones("ní hǎo", "3")


def test_parse_pinyin_and_tones_invalid_tone_type():
    with pytest.raises(ValueError, match="deben ser números enteros"):
        parse_pinyin_and_tones("ní hǎo", "3 abc")


def test_format_assessment_output():
    mock_response = {
        "utterance": "ní hǎo",
        "audio_quality": {
            "is_acceptable": True,
            "quality_score": 0.95,
            "rejection_reasons": []
        },
        "phonetic_assessment": {"overall_score": 0.90},
        "tone_assessment": {"overall_score": 0.85},
        "syllables": [
            {
                "syllable": "ní",
                "pinyin": "ní",
                "phonetics": {"overall_score": 0.92},
                "tone": {
                    "predicted_tone": 2,
                    "classification_confidence": 0.88,
                    "is_sandhi_applied": True,
                    "sandhi_rule_name": "Rule 3-3"
                },
                "overall_score": 0.90
            }
        ],
        "feedback": [
            {
                "summary": "Buena pronunciación.",
                "actionable_tip": "Mantén el tono constante.",
                "detailed_explanation": "El cambio de tono fue correcto."
            }
        ],
        "errors": []
    }

    formatted_md = format_assessment_output(mock_response)
    assert "Evaluación de Pronunciación" in formatted_md
    assert "Puntuación Fonética Global:** `90.0%`" in formatted_md
    assert "Puntuación Tonal Global:** `85.0%`" in formatted_md
    assert "Buena pronunciación." in formatted_md


def test_assess_pronunciation_ui_success():
    audio_tuple = (16000, np.zeros(16000, dtype=np.float32))
    result = assess_pronunciation_ui(audio_tuple, "ní hǎo", "3 3")
    assert "Evaluación de Pronunciación" in result
    assert "Puntuación Fonética Global" in result


def test_assess_pronunciation_ui_validation_errors():
    audio_tuple = (16000, np.zeros(16000, dtype=np.float32))
    # Tone count mismatch
    result = assess_pronunciation_ui(audio_tuple, "ní hǎo", "3")
    assert "Error de Validación de Entrada" in result

    # Missing audio
    result = assess_pronunciation_ui(None, "ní hǎo", "3 3")
    assert "Error en el Audio" in result
