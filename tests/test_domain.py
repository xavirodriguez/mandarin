from src.domain.tones.sandhi import ToneSandhiEngine
from src.domain.diagnosis.feedback import PedagogicalFeedbackGenerator
from src.domain.diagnosis.models import DiagnosticResult, ErrorCategory, ErrorType

def test_tone_sandhi_engine():
    # Test 3-3 sandhi rule
    pinyin = ["ní", "hǎo"]
    tones = [3, 3]
    results = ToneSandhiEngine.apply_sandhi_rules(pinyin, tones)
    assert results[0][0] == 2
    assert results[0][1] is True
    assert results[0][2] == "3-3_tone_sandhi"
    assert results[1][0] == 3

    # Test yi before tone 4
    pinyin = ["yī", "yàng"]
    tones = [1, 4]
    results = ToneSandhiEngine.apply_sandhi_rules(pinyin, tones)
    assert results[0][0] == 2
    assert results[0][1] is True

    # Test bu before tone 4
    pinyin = ["bù", "shì"]
    tones = [4, 4]
    results = ToneSandhiEngine.apply_sandhi_rules(pinyin, tones)
    assert results[0][0] == 2
    assert results[0][1] is True

def test_pedagogical_feedback_generator():
    diag = DiagnosticResult(
        category=ErrorCategory.PHONETIC,
        error_type=ErrorType.CONFUSION_PAIR,
        error_subtype="retroflexion_missing",
        severity=0.8,
        confidence=0.92,
        affected_syllable_idx=0,
        syllable_text="zhi",
        pinyin="zhī",
        technical_details={"target_phoneme": "zh", "detected_phoneme": "z"}
    )
    feedback = PedagogicalFeedbackGenerator.generate_feedback(diag)
    assert "Falta de retroflexión" in feedback.summary
    assert "curva la punta de la lengua" in feedback.actionable_tip.lower()
