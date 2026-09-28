import numpy as np
from typing import List, Dict, Any, Optional

from src.domain.audio.quality import AudioQualityMetrics
from src.domain.tones.sandhi import ToneSandhiEngine
from src.domain.syllables.models import SyllableAssessment
from src.domain.diagnosis.models import DiagnosticResult
from src.domain.diagnosis.feedback import PedagogicalFeedback

from src.infrastructure.audio.qa import AudioQAProcessor
from src.infrastructure.pitch.processor import PitchProcessor
from src.infrastructure.alignment.aligner import forced_aligner_mock
from src.infrastructure.models.adapters import SpeechEncoderAdapter, PhonemeRecognizerAdapter, ToneClassifierAdapter
from src.infrastructure.calibration.calibrator import ModelCalibrator

from src.application.analyze_syllable import SyllableAnalyzer
from src.application.diagnose_errors import DiagnosticEngine
from src.application.generate_feedback import FeedbackEngineService

class PronunciationAssessmentPipeline:
    """
    Main Application Orchestrator for Mandarin CAPT System.
    Executes end-to-end pipeline:
    Audio QA -> Shared Representation -> Phonetic & Tone Pipelines ->
    Syllable Integration -> Sandhi Analysis -> Error Diagnosis -> Pedagogical Feedback.
    """

    def __init__(
        self,
        audio_qa_processor: Optional[AudioQAProcessor] = None,
        speech_encoder: Optional[SpeechEncoderAdapter] = None,
        aligner: Optional[forced_aligner_mock] = None,
        syllable_analyzer: Optional[SyllableAnalyzer] = None,
        diagnostic_engine: Optional[DiagnosticEngine] = None,
        feedback_service: Optional[FeedbackEngineService] = None
    ):
        self.audio_qa = audio_qa_processor or AudioQAProcessor()
        self.speech_encoder = speech_encoder or SpeechEncoderAdapter()
        self.aligner = aligner or forced_aligner_mock()
        self.syllable_analyzer = syllable_analyzer or SyllableAnalyzer()
        self.diagnostic_engine = diagnostic_engine or DiagnosticEngine()
        self.feedback_service = feedback_service or FeedbackEngineService()

    def assess(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        target_pinyin: List[str],
        lexical_tones: List[int]
    ) -> Dict[str, Any]:
        """
        Runs complete evaluation on the input audio against reference target text and tones.
        """
        # 1. Audio QA & Verification
        audio, quality_metrics = self.audio_qa.process_and_verify(waveform, sample_rate)

        # Immediate rejection gate if audio is unacceptable
        if quality_metrics.should_reject():
            diagnostics = self.diagnostic_engine.diagnose_utterance([], quality_metrics)
            feedback = self.feedback_service.generate_feedback_reports(diagnostics)
            return {
                "utterance": " ".join(target_pinyin),
                "audio_quality": quality_metrics,
                "syllables": [],
                "phonetic_assessment": {"overall_score": 0.0},
                "tone_assessment": {"overall_score": 0.0},
                "errors": diagnostics,
                "feedback": feedback
            }

        # 2. Extract Shared Acoustic Feature Representation
        shared_features = self.speech_encoder.encode(audio, sample_rate)

        # 3. Tone Sandhi Contextual Analysis
        sandhi_results = ToneSandhiEngine.apply_sandhi_rules(target_pinyin, lexical_tones)

        # 4. Temporal Syllable Forced Alignment
        alignments = self.aligner.align(audio, sample_rate, target_pinyin)

        # 5. Syllable Assessment Analysis Loop
        syllable_assessments: List[SyllableAssessment] = []
        for idx, align in enumerate(alignments):
            lex_t = lexical_tones[idx]
            ctx_t, is_sandhi, sandhi_rule = sandhi_results[idx]

            # Slice audio segment for syllable
            start_sample = int(align["start"] * sample_rate)
            end_sample = int(align["end"] * sample_rate)
            syl_audio = audio[start_sample:end_sample] if end_sample > start_sample else audio

            syl_assessment = self.syllable_analyzer.analyze_syllable(
                audio_segment=syl_audio,
                sample_rate=sample_rate,
                alignment=align,
                lexical_tone=lex_t,
                contextual_target=ctx_t,
                audio_quality=quality_metrics,
                is_sandhi_applied=is_sandhi,
                sandhi_rule_name=sandhi_rule
            )
            syllable_assessments.append(syl_assessment)

        # 6. Multi-modal Error Diagnosis (Severity vs Confidence)
        diagnostics = self.diagnostic_engine.diagnose_utterance(syllable_assessments, quality_metrics)

        # 7. Pedagogical Feedback Generation
        feedback = self.feedback_service.generate_feedback_reports(diagnostics)

        # 8. Compute Aggregated Scores
        phonetic_scores = [s.phonetics.overall_score for s in syllable_assessments]
        avg_phonetic_score = float(np.mean(phonetic_scores)) if phonetic_scores else 0.0

        tone_scores = [s.tone.tone_probabilities.get(s.tone.contextual_target, 0.0) for s in syllable_assessments]
        avg_tone_score = float(np.mean(tone_scores)) if tone_scores else 0.0

        return {
            "utterance": " ".join(target_pinyin),
            "audio_quality": quality_metrics,
            "syllables": syllable_assessments,
            "phonetic_assessment": {"overall_score": avg_phonetic_score},
            "tone_assessment": {"overall_score": avg_tone_score},
            "errors": diagnostics,
            "feedback": feedback
        }
