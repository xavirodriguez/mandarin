"""Gradio User Interface for Mandarin CAPT System.

Provides an interactive web interface for students and teachers to evaluate
Mandarin pronunciation, phonetic accuracy, tone recognition, and receive
pedagogical feedback.

Configuration:
- Set `API_URL` environment variable to point to a custom or remote API instance.
  Example for Render: `export API_URL="https://your-app.onrender.com/api/v1/assess"`
  Default local URL: `http://localhost:8000/api/v1/assess`
"""

import os
import base64
import requests
import numpy as np
from scipy.signal import resample
import gradio as gr

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/assess")
HTTP_TIMEOUT_SEC = 120  # Extended timeout to accommodate potential server cold starts (e.g. Render)


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
    if data is None or data.size == 0:
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


def format_assessment_output(response_data: dict) -> str:
    """Format API JSON response into pedagogical Markdown output.

    Args:
        response_data: Parsed dictionary from API response.

    Returns:
        Formatted Markdown string.
    """
    utterance = response_data.get("utterance", "")
    audio_q = response_data.get("audio_quality", {})
    phonetic_q = response_data.get("phonetic_assessment", {})
    tone_q = response_data.get("tone_assessment", {})
    syllables = response_data.get("syllables", [])
    feedback = response_data.get("feedback", [])
    errors = response_data.get("errors", [])

    md = []
    md.append(f"## 🎯 Evaluación de Pronunciación: `{utterance}`\n")

    # Global Scores Card
    is_acc = audio_q.get("is_acceptable", True)
    status_icon = "✅" if is_acc else "⚠️"
    status_text = "Calidad de audio aceptable" if is_acc else "Audio con posibles problemas de calidad"

    phonetic_score = phonetic_q.get("overall_score", 0.0) * 100
    tone_score = tone_q.get("overall_score", 0.0) * 100
    audio_quality_score = audio_q.get("quality_score", 0.0) * 100

    md.append("### 📊 Puntuaciones Generales")
    md.append(f"- **Puntuación Fonética Global:** `{phonetic_score:.1f}%`")
    md.append(f"- **Puntuación Tonal Global:** `{tone_score:.1f}%`")
    md.append(f"- **Calidad de Audio:** `{audio_quality_score:.1f}%` ({status_icon} {status_text})")

    if not is_acc and audio_q.get("rejection_reasons"):
        reasons = ", ".join(audio_q["rejection_reasons"])
        md.append(f"  - *Atención:* {reasons}")

    # Syllable breakdown
    if syllables:
        md.append("\n### 🔍 Desglose por Sílaba\n")
        md.append("| Sílaba | Pinyin | Puntuación Fonética | Tono Predicho | Confianza Tono | Puntuación Total |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for s in syllables:
            s_text = s.get("syllable", "")
            s_pinyin = s.get("pinyin", "")
            p_score = s.get("phonetics", {}).get("overall_score", 0.0) * 100
            t_obj = s.get("tone", {})
            pred_tone = t_obj.get("predicted_tone", "-")
            t_conf = t_obj.get("classification_confidence", 0.0) * 100
            tot_score = s.get("overall_score", 0.0) * 100

            sandhi_note = ""
            if t_obj.get("is_sandhi_applied"):
                rule = t_obj.get("sandhi_rule_name", "Cambio de tono")
                sandhi_note = f" *(Aplica {rule})*"

            md.append(
                f"| **{s_text}** | `{s_pinyin}` | `{p_score:.1f}%` | Tono {pred_tone}{sandhi_note} | `{t_conf:.1f}%` | `{tot_score:.1f}%` |"
            )

    # Pedagogical Feedback
    if feedback:
        md.append("\n### 💡 Feedback Pedagógico\n")
        for fb in feedback:
            summary = fb.get("summary", "")
            actionable_tip = fb.get("actionable_tip", "")
            detailed = fb.get("detailed_explanation", "")

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
            cat = err.get("category", "")
            etype = err.get("error_type", "")
            pinyin_err = err.get("pinyin", "")
            sev = err.get("severity", 0.0) * 100
            md.append(f"- **Sílabas ({pinyin_err})**: Error tipo `{etype}` [{cat}] - Severidad: `{sev:.0f}%`")

    return "\n".join(md)


def assess_pronunciation_ui(audio, pinyin_input: str, tones_input: str) -> str:
    """Main Gradio event handler calling CAPT API endpoint.

    Args:
        audio: Audio input from gr.Audio (tuple or None)
        pinyin_input: String containing target pinyin.
        tones_input: String containing lexical tones.

    Returns:
        Formatted Markdown result or error message.
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

    payload = {
        "target_pinyin": target_pinyin,
        "lexical_tones": lexical_tones,
        "audio_base64": audio_base64,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=HTTP_TIMEOUT_SEC)
        if response.status_code != 200:
            try:
                err_detail = response.json().get("detail", response.text)
            except Exception:
                err_detail = response.text
            return f"❌ **Error del Servidor API ({response.status_code}):** {err_detail}"

        data = response.json()
        return format_assessment_output(data)

    except requests.exceptions.Timeout:
        return (
            f"⏳ **Tiempo de espera agotado ({HTTP_TIMEOUT_SEC}s):** "
            f"El servidor en `{API_URL}` tardó demasiado en responder. "
            "Si está desplegado en Render u otro servicio gratuito, puede estar despertando de un arranque en frío (cold start). "
            "Por favor, reintenta en unos segundos."
        )
    except requests.exceptions.ConnectionError:
        return (
            f"🚫 **No se pudo conectar con la API en `{API_URL}`:** "
            "Asegúrate de que el servidor FastAPI está en ejecución. "
            "Puedes iniciarlo con:\n"
            "```bash\n"
            "uvicorn src.interfaces.api.routes:app --host 0.0.0.0 --port 8000\n"
            "```"
        )
    except Exception as e:
        return f"❌ **Error inesperado:** {str(e)}"


def build_app():
    """Build Gradio Blocks application."""
    theme = gr.themes.Soft()

    with gr.Blocks(title="Mandarin CAPT - Evaluación de Pronunciación") as demo:
        gr.Markdown(
            """
            # 🇨🇳 Mandarin CAPT — Interfaz de Evaluación de Pronunciación
            Evalúa tu pronunciación en mandarín, recibe feedback fonético y tonal detallado con diagnóstico pedagógico.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="🎙️ Graba o sube tu pronunciación",
                )
                pinyin_input = gr.Textbox(
                    label="📝 Pinyin Objetivo",
                    placeholder="Ejemplo: nǐ hǎo o ni3 hao3",
                    value="nǐ hǎo",
                )
                tones_input = gr.Textbox(
                    label="🎵 Tonos Léxicos",
                    placeholder="Ejemplo: 3 3",
                    value="3 3",
                )

                submit_btn = gr.Button("🔍 Evaluar Pronunciación", variant="primary")

                gr.Markdown(
                    f"""
                    ---
                    ℹ️ **Configuración del Servidor API:**
                    * URL de API actual: `{API_URL}`
                    * Para conectar la UI a una API remota (ejemplo Render), define la variable de entorno:
                      `export API_URL="https://tu-app.onrender.com/api/v1/assess"`
                    """
                )

            with gr.Column(scale=1):
                output_markdown = gr.Markdown(
                    value="👉 *Selecciona o graba tu audio y pulsa 'Evaluar Pronunciación' para ver los resultados.*",
                    label="Resultados de la Evaluación",
                )

        gr.Examples(
            examples=[
                ["nǐ hǎo", "3 3"],
                ["xiè xie", "4 0"],
                ["zài jiàn", "4 4"],
            ],
            inputs=[pinyin_input, tones_input],
            label="💡 Ejemplos pre-cargados",
        )

        submit_btn.click(
            fn=assess_pronunciation_ui,
            inputs=[audio_input, pinyin_input, tones_input],
            outputs=[output_markdown],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(theme=gr.themes.Soft())
