import io
import base64
import json
from typing import Tuple
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
import numpy as np
import soundfile as sf
import librosa

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

def decode_audio_bytes(raw_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decodes audio bytes (WAV or raw PCM) into a float32 1D numpy array at 16kHz.
    Validates minimum and maximum audio duration.
    """
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio payload is empty."
        )

    # 1. Decode audio using soundfile or fallback to raw int16 PCM
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # convert stereo to mono
        audio = data
    except Exception:
        try:
            pcm16 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio = pcm16
            sr = 16000
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not decode audio payload: {str(e)}"
            )

    if audio is None or len(audio) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Decoded audio waveform is empty."
        )

    # 2. Resample to 16 kHz if sample rate differs
    target_sr = 16000
    if sr != target_sr:
        try:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to resample audio from {sr} Hz to {target_sr} Hz: {str(e)}"
            )

    # 3. Validate duration bounds (0.1s to 30.0s)
    duration_sec = len(audio) / float(sr)
    MIN_DURATION_SEC = 0.1
    MAX_DURATION_SEC = 30.0

    if duration_sec < MIN_DURATION_SEC or duration_sec > MAX_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio duration ({duration_sec:.2f}s) is out of valid bounds [{MIN_DURATION_SEC}s, {MAX_DURATION_SEC}s]."
        )

    return audio, sr

def _build_assessment_response(result: dict) -> AssessmentResponseSchema:
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

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Mandarin CAPT System"}

@app.post("/api/v1/assess", response_model=AssessmentResponseSchema)
def assess_pronunciation(request: AssessmentRequestSchema):
    if not request.audio_base64 or not request.audio_base64.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio is required: audio_base64 parameter missing or empty."
        )

    try:
        raw_bytes = base64.b64decode(request.audio_base64)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 audio string: {str(e)}"
        )

    audio, sr = decode_audio_bytes(raw_bytes)

    result = pipeline.assess(
        waveform=audio,
        sample_rate=sr,
        target_pinyin=request.target_pinyin,
        lexical_tones=request.lexical_tones
    )

    return _build_assessment_response(result)

@app.post("/api/v1/assess/upload", response_model=AssessmentResponseSchema)
async def assess_pronunciation_file(
    target_pinyin: str = Form(..., description="JSON encoded list of pinyin syllables e.g. '[\"ní\", \"hǎo\"]'"),
    lexical_tones: str = Form(..., description="JSON encoded list of tone integers e.g. '[3, 3]'"),
    file: UploadFile = File(...)
):
    try:
        pinyin_list = json.loads(target_pinyin)
        tones_list = json.loads(lexical_tones)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON parameters: {str(e)}"
        )

    raw_bytes = await file.read()
    audio, sr = decode_audio_bytes(raw_bytes)

    result = pipeline.assess(
        waveform=audio,
        sample_rate=sr,
        target_pinyin=pinyin_list,
        lexical_tones=tones_list
    )

    return _build_assessment_response(result)
