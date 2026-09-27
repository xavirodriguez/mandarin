from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict

class ErrorCategory(Enum):
    PHONETIC = "phonetic"
    TONE = "tone"
    DURATION = "duration"
    AUDIO_QUALITY = "audio_quality"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

class ErrorType(Enum):
    # Phonetic errors
    SUBSTITUTION = "substitution"
    OMISSION = "omission"
    INSERTION = "insertion"
    RETROFLEXION_MISSING = "retroflexion_missing"
    ASPIRATION_ERROR = "aspiration_error"
    FRICATION_ERROR = "frication_error"
    VOWEL_DISTORTION = "vowel_distortion"
    CONFUSION_PAIR = "confusion_pair"

    # Tone errors
    TONE_MISCLASSIFICATION = "tone_misclassification"
    INSUFFICIENT_FALL = "insufficient_fall"
    INSUFFICIENT_RISE = "insufficient_rise"
    INSUFFICIENT_DIP = "insufficient_dip"
    FLAT_CONTOUR = "flat_contour"
    PITCH_REGISTER_TOO_HIGH = "pitch_register_too_high"
    PITCH_REGISTER_TOO_LOW = "pitch_register_too_low"

    # Audio errors
    HIGH_NOISE = "high_noise"
    CLIPPING = "clipping"
    TOO_SHORT = "too_short"

    UNKNOWN = "unknown"

class ErrorSeverity(Enum):
    MINOR = "minor"        # 0.1 - 0.4
    MODERATE = "moderate"  # 0.4 - 0.7
    CRITICAL = "critical"  # 0.7 - 1.0

@dataclass
class DiagnosticResult:
    category: ErrorCategory
    error_type: ErrorType
    error_subtype: str
    severity: float         # 0.0 to 1.0 (estimated severity of pronunciation error)
    confidence: float       # 0.0 to 1.0 (model confidence in diagnosis)
    affected_syllable_idx: int
    syllable_text: str
    pinyin: str
    technical_details: Dict[str, str]
