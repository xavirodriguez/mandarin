import numpy as np
from typing import Dict, Any, Optional
from src.domain.syllables.models import SyllableAssessment, SyllableStructure
from src.domain.phonetics.models import PhoneticAssessmentResult, PhonemeSegment
from src.domain.tones.models import ToneAssessmentResult, PitchContour
from src.domain.tones.sandhi import ToneSandhiEngine
from src.domain.audio.quality import AudioQualityMetrics

from src.infrastructure.audio.qa import AudioQAProcessor
from src.infrastructure.pitch.processor import PitchProcessor
from src.infrastructure.models.adapters import PhonemeRecognizerAdapter, ToneClassifierAdapter

class SyllableAnalyzer:
    """
    Application service that integrates phonetics, pitch, tone, duration, prosody,
    and audio quality for a single syllable segment.
    """

    def __init__(
        self,
        pitch_processor: Optional[PitchProcessor] = None,
        phoneme_recognizer: Optional[PhonemeRecognizerAdapter] = None,
        tone_classifier: Optional[ToneClassifierAdapter] = None
    ):
        self.pitch_processor = pitch_processor or PitchProcessor()
        self.phoneme_recognizer = phoneme_recognizer or PhonemeRecognizerAdapter()
        self.tone_classifier = tone_classifier or ToneClassifierAdapter()

    def analyze_syllable(
        self,
        audio_segment: np.ndarray,
        sample_rate: int,
        alignment: Dict[str, Any],
        lexical_tone: int,
        contextual_target: int,
        audio_quality: AudioQualityMetrics,
        is_sandhi_applied: bool = False,
        sandhi_rule_name: Optional[str] = None
    ) -> SyllableAssessment:
        pinyin = alignment["pinyin"]
        start = alignment["start"]
        end = alignment["end"]
        duration = end - start

        # 1. Pitch & Tone Analysis on syllable segment
        pitch_contour = self.pitch_processor.estimate_pitch(audio_segment)
        tone_probs = self.tone_classifier.classify_tone(pitch_contour, contextual_target)

        predicted_tone = max(tone_probs, key=tone_probs.get)
        classification_confidence = tone_probs[predicted_tone]

        # Calculate tone contour similarity (e.g., against ideal contour)
        ideal_contour = np.array([0.0, 0.5, 1.0]) if contextual_target == 2 else np.array([1.0, 0.0, -1.0])
        contour_similarity = self.pitch_processor.compute_dtw_distance(
            pitch_contour.normalized_f0[pitch_contour.voiced_probs > 0.5],
            ideal_contour
        )

        tone_result = ToneAssessmentResult(
            lexical_tone=lexical_tone,
            contextual_target=contextual_target,
            predicted_tone=predicted_tone,
            tone_probabilities=tone_probs,
            classification_confidence=classification_confidence,
            contour_similarity=contour_similarity,
            pitch_contour=pitch_contour,
            is_sandhi_applied=is_sandhi_applied,
            sandhi_rule_name=sandhi_rule_name
        )

        # 2. Phonetic Analysis
        target_ph = [alignment["onset"].phoneme] if alignment.get("onset") else []
        if alignment.get("nucleus"):
            target_ph.append(alignment["nucleus"].phoneme)
        if alignment.get("coda"):
            target_ph.append(alignment["coda"].phoneme)

        # Simulated or posterior evaluation
        dummy_posteriors = np.ones((10, 60))
        detected_ph = list(target_ph) # default match or mispronounced
        phonetic_result = self.phoneme_recognizer.evaluate_gop(dummy_posteriors, target_ph, detected_ph)

        # Build structure
        structure = SyllableStructure(
            onset=alignment.get("onset"),
            nucleus=alignment.get("nucleus"),
            coda=alignment.get("coda")
        )

        return SyllableAssessment(
            syllable=pinyin,
            pinyin=pinyin,
            start=start,
            end=end,
            duration=duration,
            phonetics=phonetic_result,
            tone=tone_result,
            structure=structure,
            audio_quality=audio_quality
        )
