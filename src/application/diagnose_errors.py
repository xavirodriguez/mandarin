from typing import List, Optional
from src.domain.diagnosis.models import (
    DiagnosticResult, ErrorCategory, ErrorType, ErrorSeverity
)
from src.domain.syllables.models import SyllableAssessment
from src.domain.audio.quality import AudioQualityMetrics
from src.infrastructure.calibration.calibrator import ModelCalibrator

class DiagnosticEngine:
    """
    Diagnostic Engine that combines Phonetic Assessment, Alignment, Pitch/F0 Analysis,
    Tone Classification, Prosodic Context/Sandhi, Audio Quality, and Model Confidence.
    Generates explicit DiagnosticResults distinguishing severity vs confidence.
    """

    def __init__(self, calibrator: Optional[ModelCalibrator] = None):
        self.calibrator = calibrator or ModelCalibrator()

    def diagnose_utterance(
        self,
        syllable_assessments: List[SyllableAssessment],
        audio_quality: AudioQualityMetrics
    ) -> List[DiagnosticResult]:
        diagnostic_results: List[DiagnosticResult] = []

        # 1. Reject or flag low audio quality
        if not audio_quality.is_acceptable:
            diagnostic_results.append(DiagnosticResult(
                category=ErrorCategory.AUDIO_QUALITY,
                error_type=ErrorType.HIGH_NOISE if "SNR" in str(audio_quality.rejection_reasons) else ErrorType.CLIPPING,
                error_subtype="unacceptable_audio_quality",
                severity=1.0,
                confidence=0.99,
                affected_syllable_idx=-1,
                syllable_text="",
                pinyin="",
                technical_details={"reasons": "; ".join(audio_quality.rejection_reasons)}
            ))
            return diagnostic_results

        # 2. Syllable-by-syllable diagnostic analysis
        for idx, syl in enumerate(syllable_assessments):
            syl_quality = syl.audio_quality.quality_score

            # A. Check Phonetic Errors
            if syl.phonetics.substitutions or syl.phonetics.overall_score < 0.70:
                for target_p, det_p in syl.phonetics.substitutions:
                    is_confusion = syl.phonetics.confusion_type == "mandarin_confusion_pair"
                    err_type = ErrorType.CONFUSION_PAIR if is_confusion else ErrorType.SUBSTITUTION

                    if target_p in ("zh", "ch", "sh") and det_p in ("z", "c", "s"):
                        err_type = ErrorType.RETROFLEXION_MISSING

                    severity = float(min(1.0, (1.0 - syl.phonetics.overall_score) * 1.2))
                    confidence = float(syl.phonetics.confidence * syl_quality)

                    if not self.calibrator.is_evidence_sufficient(confidence, syl_quality):
                        diagnostic_results.append(self._create_insufficient_evidence(idx, syl))
                    else:
                        diagnostic_results.append(DiagnosticResult(
                            category=ErrorCategory.PHONETIC,
                            error_type=err_type,
                            error_subtype=err_type.value,
                            severity=severity,
                            confidence=confidence,
                            affected_syllable_idx=idx,
                            syllable_text=syl.syllable,
                            pinyin=syl.pinyin,
                            technical_details={"target_phoneme": target_p, "detected_phoneme": det_p}
                        ))

            # B. Check Tone Errors
            tone = syl.tone
            contextual_target = tone.contextual_target
            pred_tone = tone.predicted_tone
            tone_prob = tone.tone_probabilities.get(contextual_target, 0.0)

            if pred_tone != contextual_target or tone_prob < 0.60:
                tone_confidence = float(tone.classification_confidence * syl_quality)
                tone_severity = float(min(1.0, (1.0 - tone_prob) * 1.1))

                # Identify specific tone contour defect
                slope = tone.pitch_contour.slope
                err_type = ErrorType.TONE_MISCLASSIFICATION

                if contextual_target == 4 and tone.pitch_contour.slope > -10.0:
                    err_type = ErrorType.INSUFFICIENT_FALL
                elif contextual_target == 2 and tone.pitch_contour.slope < 10.0:
                    err_type = ErrorType.INSUFFICIENT_RISE
                elif contextual_target == 3 and tone.contour_similarity < 0.4:
                    err_type = ErrorType.INSUFFICIENT_DIP

                if not self.calibrator.is_evidence_sufficient(tone_confidence, syl_quality):
                    diagnostic_results.append(self._create_insufficient_evidence(idx, syl))
                else:
                    diagnostic_results.append(DiagnosticResult(
                        category=ErrorCategory.TONE,
                        error_type=err_type,
                        error_subtype=err_type.value,
                        severity=tone_severity,
                        confidence=tone_confidence,
                        affected_syllable_idx=idx,
                        syllable_text=syl.syllable,
                        pinyin=syl.pinyin,
                        technical_details={
                            "target_tone": str(contextual_target),
                            "predicted_tone": str(pred_tone),
                            "sandhi_applied": str(tone.is_sandhi_applied)
                        }
                    ))

        return diagnostic_results

    def _create_insufficient_evidence(self, idx: int, syl: SyllableAssessment) -> DiagnosticResult:
        return DiagnosticResult(
            category=ErrorCategory.INSUFFICIENT_EVIDENCE,
            error_type=ErrorType.UNKNOWN,
            error_subtype="insufficient_evidence",
            severity=0.0,
            confidence=0.30,
            affected_syllable_idx=idx,
            syllable_text=syl.syllable,
            pinyin=syl.pinyin,
            technical_details={"reason": "Low signal quality or low model classification confidence"}
        )
