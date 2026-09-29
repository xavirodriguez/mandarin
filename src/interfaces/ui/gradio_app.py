"""Gradio User Interface for Mandarin CAPT System.

Provides an interactive web interface for students and teachers to evaluate
Mandarin pronunciation, phonetic accuracy, tone recognition, and receive
pedagogical feedback.
"""

import base64
import numpy as np
from scipy.signal import resample
import gradio as gr


def preprocess_audio(audio_tuple) -> str:
    """Preprocess input audio array to 16 kHz mono int16 raw PCM base64 string.

    Args:
        audio_tuple: Tuple of (sample_rate, audio_data_numpy) from gr.Audio(type="numpy")

    Returns:
        Base64-encoded PCM int16 audio string.

    Raises:
        ValueError: If audio structure or data is invalid/empty.
    """
    if audio_tuple is None:
        raise ValueError("No se proporcionó ningún archivo o grabación de audio.")

    orig_sr, data = audio_tuple
    if data is None or getattr(data, "size", 0) == 0:
        raise ValueError("El audio proporcionado está vacío.")

    # Convert to float32
    audio = data.astype(np.float32)

    # Convert multi-channel (stereo) to mono
    if audio.ndim > 1:
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T
        audio = np.mean(audio, axis=1)

    # Normalize float values to range [-1.0, 1.0] if needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

    # Resample to 16000 Hz if necessary
    target_sr = 16000
    if orig_sr != target_sr and len(audio) > 0:
        num_samples = int(round(len(audio) * target_sr / orig_sr))
        audio = resample(audio, num_samples)

    # Scale float32 [-1, 1] to PCM int16 [-32768, 32767]
    pcm_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

    # Base64 encode raw PCM bytes
    raw_bytes = pcm_int16.tobytes()
    return base64.b64encode(raw_bytes).decode("utf-8")


def parse_pinyin_and_tones(pinyin_str: str, tones_str: str):
    """Parse and validate pinyin and tone string inputs.

    Args:
        pinyin_str: Space or comma-separated target pinyin syllables (e.g., 'ní hǎo' or 'ni3 hao3').
        tones_str: Space or comma-separated lexical tones (e.g., '3 3').

    Returns:
        Tuple of (list of pinyin strings, list of int tones).

    Raises:
        ValueError: If parsing fails or length mismatch occurs.
    """
    clean_pinyin = [p.strip() for p in pinyin_str.replace(",", " ").split() if p.strip()]
    if not clean_pinyin:
        raise ValueError("Por favor introduce al menos una sílaba Pinyin objetivo.")

    raw_tones = [t.strip() for t in tones_str.replace(",", " ").split() if t.strip()]
    if not raw_tones:
        raise ValueError("Por favor introduce los tonos léxicos correspondientes.")

    try:
        clean_tones = [int(t) for t in raw_tones]
    except ValueError:
        raise ValueError("Los tonos deben ser números enteros separados por espacios (ejemplo: 3 3).")

    if len(clean_pinyin) != len(clean_tones):
        raise ValueError(
            f"El número de sílabas Pinyin ({len(clean_pinyin)}) no coincide con el número de tonos ({len(clean_tones)})."
        )

    return clean_pinyin, clean_tones


def _get_val(obj, key, default=None):
    """Helper to access dict or object attributes uniformly."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def format_assessment_output(response_data: dict) -> str:
    """Format assessment pipeline output into pedagogical Markdown.

    Args:
        response_data: Dict or pipeline result containing scores, quality, syllables, feedback.

    Returns:
        Formatted Markdown string.
    """
    utterance = _get_val(response_data, "utterance", "")
    audio_q = _get_val(response_data, "audio_quality", {})
    phonetic_q = _get_val(response_data, "phonetic_assessment", {})
    tone_q = _get_val(response_data, "tone_assessment", {})
    syllables = _get_val(response_data, "syllables", [])
    feedback = _get_val(response_data, "feedback", [])
    errors = _get_val(response_data, "errors", [])

    md = []
    md.append(f"## 🎯 Evaluación de Pronunciación: `{utterance}`\n")

    # Global Scores Card
    is_acc = _get_val(audio_q, "is_acceptable", True)
    status_icon = "✅" if is_acc else "⚠️"
    status_text = "Calidad de audio aceptable" if is_acc else "Audio con posibles problemas de calidad"

    phonetic_score = _get_val(phonetic_q, "overall_score", 0.0) * 100
    tone_score = _get_val(tone_q, "overall_score", 0.0) * 100
    audio_quality_score = _get_val(audio_q, "quality_score", 0.0) * 100

    md.append("### 📊 Puntuaciones Generales")
    md.append(f"- **Puntuación Fonética Global:** `{phonetic_score:.1f}%`")
    md.append(f"- **Puntuación Tonal Global:** `{tone_score:.1f}%`")
    md.append(f"- **Calidad de Audio:** `{audio_quality_score:.1f}%` ({status_icon} {status_text})")

    rejection_reasons = _get_val(audio_q, "rejection_reasons", [])
    if not is_acc and rejection_reasons:
        reasons = ", ".join(rejection_reasons)
        md.append(f"  - *Atención:* {reasons}")

    # Syllable breakdown
    if syllables:
        md.append("\n### 🔍 Desglose por Sílaba\n")
        md.append("| Sílaba | Pinyin | Puntuación Fonética | Tono Predicho | Confianza Tono | Puntuación Total |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for s in syllables:
            s_text = _get_val(s, "syllable", "")
            s_pinyin = _get_val(s, "pinyin", "")
            phonetics_obj = _get_val(s, "phonetics", {})
            p_score = _get_val(phonetics_obj, "overall_score", 0.0) * 100

            tone_obj = _get_val(s, "tone", {})
            pred_tone = _get_val(tone_obj, "predicted_tone", "-")
            t_conf = _get_val(tone_obj, "classification_confidence", 0.0) * 100
            tot_score = _get_val(s, "overall_score", 0.0) * 100

            sandhi_note = ""
            if _get_val(tone_obj, "is_sandhi_applied", False):
                rule = _get_val(tone_obj, "sandhi_rule_name", "Cambio de tono")
                sandhi_note = f" *(Aplica {rule})*"

            md.append(
                f"| **{s_text}** | `{s_pinyin}` | `{p_score:.1f}%` | Tono {pred_tone}{sandhi_note} | `{t_conf:.1f}%` | `{tot_score:.1f}%` |"
            )

    # Pedagogical Feedback
    if feedback:
        md.append("\n### 💡 Feedback Pedagógico\n")
        for fb in feedback:
            summary = _get_val(fb, "summary", "")
            actionable_tip = _get_val(fb, "actionable_tip", "")
            detailed = _get_val(fb, "detailed_explanation", "")

            if summary:
                md.append(f"**Resumen:** {summary}\n")
            if actionable_tip:
                md.append(f"**Consejo Práctico:** {actionable_tip}\n")
            if detailed:
                md.append(f"**Explicación Detallada:** {detailed}\n")

    # Diagnostic Errors
    if errors:
        md.append("\n### 🚨 Diagnóstico de Errores\n")
        for err in errors:
            cat = _get_val(err, "category", "")
            etype = _get_val(err, "error_type", "")
            pinyin_err = _get_val(err, "pinyin", "")
            sev = _get_val(err, "severity", 0.0) * 100
            md.append(f"- **Sílabas ({pinyin_err})**: Error tipo `{etype}` [{cat}] - Severidad: `{sev:.0f}%`")

    return "\n".join(md)


def assess_pronunciation_ui(audio, pinyin_input: str, tones_input: str) -> str:
    """Main Gradio event handler calling local CAPT assessment pipeline.

    Args:
        audio: Audio input tuple from gr.Audio(type="numpy") or None.
        pinyin_input: String containing target pinyin.
        tones_input: String containing lexical tones.

    Returns:
        Formatted Markdown result or error message string.
    """
    try:
        target_pinyin, lexical_tones = parse_pinyin_and_tones(pinyin_input, tones_input)
    except ValueError as ve:
        return f"⚠️ **Error de Validación de Entrada:** {str(ve)}"

    try:
        audio_base64 = preprocess_audio(audio)
    except ValueError as ve:
        return f"⚠️ **Error en el Audio:** {str(ve)}"
    except Exception as e:
        return f"❌ **Error al procesar el audio:** {str(e)}"

    try:
        raw_bytes = base64.b64decode(audio_base64)
        waveform = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        from src.interfaces.api.routes import pipeline

        result = pipeline.assess(
            waveform=waveform,
            sample_rate=16000,
            target_pinyin=target_pinyin,
            lexical_tones=lexical_tones,
        )

        return format_assessment_output(result)
    except Exception as e:
        return f"❌ **Error durante la evaluación:** {str(e)}"


# Define Interface demo object
demo = gr.Interface(
    fn=assess_pronunciation_ui,
    inputs=[
        gr.Audio(sources=["microphone"], type="numpy", label="🎙️ Graba tu pronunciación"),
        gr.Textbox(
            label="📝 Pinyin Objetivo",
            placeholder="Ejemplo: ní hǎo o ni3 hao3",
            value="ní hǎo",
        ),
        gr.Textbox(
            label="🎵 Tonos Léxicos",
            placeholder="Ejemplo: 3 3",
            value="3 3",
        ),
    ],
    outputs=gr.Markdown(label="Resultados de la Evaluación"),
    title="Mandarin CAPT — Evaluación de Pronunciación",
    description="Sistema CAPT de evaluación de pronunciación en mandarín. Graba tu voz e ingresa el Pinyin objetivo y tonos léxicos.",
)
