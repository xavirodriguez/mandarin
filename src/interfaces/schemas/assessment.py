"""Pydantic schemas for assessment requests and API responses."""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class AudioQualitySchema(BaseModel):
    """Audio quality metrics schema."""
    snr_db: float
    clipping_ratio: float
    speech_duration_sec: float
    total_duration_sec: float
    voiced_ratio: float
    is_acceptable: bool
    rejection_reasons: List[str]
    quality_score: float

class PhonemeSegmentSchema(BaseModel):
    phoneme: str
    start: float
    end: float
    confidence: float

class PhoneticAssessmentSchema(BaseModel):
    target_phonemes: List[str]
    detected_phonemes: List[str]
    phoneme_scores: Dict[str, float]
    overall_score: float
    confidence: float
    substitutions: List[List[str]] = [] # [target, detected]
    confusion_type: Optional[str] = None

class ToneAssessmentSchema(BaseModel):
    lexical_tone: int
    contextual_target: int
    predicted_tone: int
    tone_probabilities: Dict[int, float]
    classification_confidence: float
    contour_similarity: float
    is_sandhi_applied: bool
    sandhi_rule_name: Optional[str] = None

class SyllableAssessmentSchema(BaseModel):
    syllable: str
    pinyin: str
    start: float
    end: float
    duration: float
    phonetics: PhoneticAssessmentSchema
    tone: ToneAssessmentSchema
    overall_score: float

class ErrorDiagnosticSchema(BaseModel):
    category: str
    error_type: str
    error_subtype: str
    severity: float = Field(..., description="Estimated severity of the error (0.0 to 1.0)")
    confidence: float = Field(..., description="Model confidence in diagnosis (0.0 to 1.0)")
    affected_syllable_idx: int
    syllable_text: str
    pinyin: str
    technical_details: Dict[str, str]

class PedagogicalFeedbackSchema(BaseModel):
    summary: str
    actionable_tip: str
    detailed_explanation: str
    target_pinyin: str
    student_pinyin: str

class AssessmentRequestSchema(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user ID for persisting assessment results")
    target_pinyin: List[str] = Field(..., json_schema_extra={"example": ["ní", "hǎo"]})
    lexical_tones: List[int] = Field(..., json_schema_extra={"example": [3, 3]})
    audio_base64: Optional[str] = None

class AssessmentResponseSchema(BaseModel):
    utterance: str
    assessment_id: Optional[str] = Field(None, description="Persisted record ID if user_id was supplied")
    audio_quality: AudioQualitySchema
    syllables: List[SyllableAssessmentSchema]
    phonetic_assessment: Dict[str, float]
    tone_assessment: Dict[str, float]
    errors: List[ErrorDiagnosticSchema]
    feedback: List[PedagogicalFeedbackSchema]
