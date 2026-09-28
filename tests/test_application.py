import numpy as np
from src.application.assess_pronunciation import PronunciationAssessmentPipeline

def test_full_application_pipeline():
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 300 * t)

    pipeline = PronunciationAssessmentPipeline()
    pinyin = ["ní", "hǎo"]
    tones = [3, 3] # Should trigger 3-3 tone sandhi -> [2, 3]

    result = pipeline.assess(audio, sr, pinyin, tones)

    assert result["utterance"] == "ní hǎo"
    assert len(result["syllables"]) == 2

    # Verify Sandhi was applied to syllable 0
    syl0 = result["syllables"][0]
    assert syl0.tone.contextual_target == 2
    assert syl0.tone.is_sandhi_applied is True
    assert syl0.tone.sandhi_rule_name == "3-3_tone_sandhi"

    # Verify structured feedback is generated
    assert isinstance(result["feedback"], list)

def test_pipeline_length_mismatch_guard(caplog):
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 300 * t)

    pipeline = PronunciationAssessmentPipeline()

    # Pass 2 pinyin syllables but 3 lexical tones
    pinyin = ["ní", "hǎo"]
    tones = [3, 3, 1]

    result = pipeline.assess(audio, sr, pinyin, tones)

    # Truncated to minimum length 2 without IndexError
    assert len(result["syllables"]) == 2
    assert "Length mismatch in pipeline assessment" in caplog.text
