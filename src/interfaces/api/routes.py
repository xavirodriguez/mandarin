from fastapi import FastAPI, HTTPException, status
import numpy as np
import base64

from src.interfaces.schemas.assessment import (
    AssessmentRequestSchema, AssessmentResponseSchema,
    AudioQualitySchema, SyllableAssessmentSchema, PhoneticAssessmentSchema,
    ToneAssessmentSchema, ErrorDiagnosticSchema, PedagogicalFeedbackSchema
)
from src.application.assess_pronunciation import PronunciationAssessmentPipeline
from src.infrastructure.persistence.assessment_repository import SQLAssessmentRepository

app = FastAPI(
    title="Mandarin CAPT Pronunciation Assessment API",
    description="Production CAPT API for Mandarin Phonetic & Tone Evaluation with Uncertainty-aware Diagnostics and Pedagogical Feedback",
    version="1.0.0"
)

pipeline = PronunciationAssessmentPipeline()
repository = SQLAssessmentRepository()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Mandarin CAPT System"}

@app.post("/api/v1/assess", response_model=AssessmentResponseSchema)
def assess_pronunciation(request: AssessmentRequestSchema):
    # Decode audio if provided or generate synthetic audio for demo/test
    if request.audio_base64:
        try:
            raw_bytes = base64.b64decode(request.audio_base64)
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 audio payload: {str(e)}"
            )
    else:
        # Generate synthetic 1 sec waveform for demonstration API calls
        sr = 16000
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 260 * t)

    result = pipeline.assess(
        waveform=audio,
        sample_rate=16000,
        target_pinyin=request.target_pinyin,
        lexical_tones=request.lexical_tones
    )

    # Serialize results to Pydantic schemas
    audio_q = result["audio_quality"]
    audio_q_schema = AudioQualitySchema(
        snr_db=audio_q.snr_db,
        clipping_ratio=audio_q.clipping_ratio,
        speech_duration_sec=audio_q.speech_duration_sec,
        total_duration_sec=audio_q.total_duration_sec,
        voiced_ratio=audio_q.voiced_ratio,
        is_acceptable=audio_q.is_acceptable,
        rejection_reasons=audio_q.rejection_reasons,
        quality_score=audio_q.quality_score
    )

    syllable_schemas = []
    for s in result["syllables"]:
        p_schema = PhoneticAssessmentSchema(
            target_phonemes=s.phonetics.target_phonemes,
            detected_phonemes=s.phonetics.detected_phonemes,
            phoneme_scores=s.phonetics.phoneme_scores,
            overall_score=s.phonetics.overall_score,
            confidence=s.phonetics.confidence,
            substitutions=[[sub[0], sub[1]] for sub in s.phonetics.substitutions],
            confusion_type=s.phonetics.confusion_type
        )
        t_schema = ToneAssessmentSchema(
            lexical_tone=s.tone.lexical_tone,
            contextual_target=s.tone.contextual_target,
            predicted_tone=s.tone.predicted_tone,
            tone_probabilities=s.tone.tone_probabilities,
            classification_confidence=s.tone.classification_confidence,
            contour_similarity=s.tone.contour_similarity,
            is_sandhi_applied=s.tone.is_sandhi_applied,
            sandhi_rule_name=s.tone.sandhi_rule_name
        )
        syllable_schemas.append(SyllableAssessmentSchema(
            syllable=s.syllable,
            pinyin=s.pinyin,
            start=s.start,
            end=s.end,
            duration=s.duration,
            phonetics=p_schema,
            tone=t_schema,
            overall_score=s.overall_score
        ))

    error_schemas = [
        ErrorDiagnosticSchema(
            category=err.category.value,
            error_type=err.error_type.value,
            error_subtype=err.error_subtype,
            severity=err.severity,
            confidence=err.confidence,
            affected_syllable_idx=err.affected_syllable_idx,
            syllable_text=err.syllable_text,
            pinyin=err.pinyin,
            technical_details=err.technical_details
        )
        for err in result["errors"]
    ]

    feedback_schemas = [
        PedagogicalFeedbackSchema(
            summary=fb.summary,
            actionable_tip=fb.actionable_tip,
            detailed_explanation=fb.detailed_explanation,
            target_pinyin=fb.target_pinyin,
            student_pinyin=fb.student_pinyin
        )
        for fb in result["feedback"]
    ]

    assessment_id = None
    if request.user_id:
        assessment_id = repository.save_assessment(result, user_id=request.user_id)

    return AssessmentResponseSchema(
        utterance=result["utterance"],
        assessment_id=assessment_id,
        audio_quality=audio_q_schema,
        syllables=syllable_schemas,
        phonetic_assessment=result["phonetic_assessment"],
        tone_assessment=result["tone_assessment"],
        errors=error_schemas,
        feedback=feedback_schemas
    )
