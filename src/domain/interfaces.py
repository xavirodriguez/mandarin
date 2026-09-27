from typing import Protocol, List, Dict, Tuple, Optional, Any
import numpy as np

from src.domain.audio.quality import AudioQualityMetrics
from src.domain.phonetics.models import PhoneticAssessmentResult, PhonemeSegment
from src.domain.tones.models import ToneAssessmentResult, PitchContour
from src.domain.syllables.models import SyllableAssessment
from src.domain.diagnosis.models import DiagnosticResult
from src.domain.diagnosis.feedback import PedagogicalFeedback

class SpeechEncoder(Protocol):
    """Protocol for Self-Supervised Speech Representation Encoder (e.g. Wav2Vec2, HuBERT, WavLM)."""
    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Transforms raw audio waveform into acoustic feature representations."""
        ...

class PhonemeRecognizer(Protocol):
    """Protocol for Phoneme Probability and Feature Extractor."""
    def recognize_posteriors(self, speech_representation: np.ndarray) -> np.ndarray:
        """Returns frame-level phoneme posterior probability distribution matrix."""
        ...

    def compute_acoustic_embedding_similarity(self, segment_audio: np.ndarray, target_phoneme: str) -> float:
        """Computes cosine similarity between target phoneme reference embedding and candidate segment."""
        ...

class PhonemeAligner(Protocol):
    """Protocol for Forced Alignment of Utterances into Syllables and Phonemes."""
    def align(self, audio: np.ndarray, sample_rate: int, target_pinyin: List[str]) -> List[Dict[str, Any]]:
        """
        Returns temporal boundary alignments for each syllable:
        onset, nucleus, coda time bounds and confidence.
        """
        ...

class PitchEstimator(Protocol):
    """Protocol for F0 pitch extraction and voiced probability estimation."""
    def estimate_pitch(self, audio: np.ndarray, sample_rate: int) -> PitchContour:
        """Estimates F0 contour, voiced probability, and time stamps."""
        ...

class ToneClassifier(Protocol):
    """Protocol for Tone Classification given Pitch Contour and Syllable Segment."""
    def classify_tone(self, contour: PitchContour, contextual_target_tone: int) -> Dict[int, float]:
        """Returns posterior probability distribution over Mandarin tones {1, 2, 3, 4, 0}."""
        ...

class PronunciationAssessor(Protocol):
    """Protocol for Goodness of Pronunciation (GOP) and phonetic evaluation."""
    def evaluate_phonetics(
        self,
        posteriors: np.ndarray,
        target_phonemes: List[str],
        start_frame: int,
        end_frame: int
    ) -> PhoneticAssessmentResult:
        ...

class ErrorDiagnoser(Protocol):
    """Protocol for combining multi-modal assessment features into explicit diagnostic results."""
    def diagnose(
        self,
        syllable_assessments: List[SyllableAssessment],
        overall_audio_quality: AudioQualityMetrics
    ) -> List[DiagnosticResult]:
        ...

class FeedbackGenerator(Protocol):
    """Protocol for converting technical diagnostic results into actionable pedagogical feedback."""
    def generate(self, diagnostic_results: List[DiagnosticResult]) -> List[PedagogicalFeedback]:
        ...
