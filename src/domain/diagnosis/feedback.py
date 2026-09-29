# pylint: disable=too-many-return-statements,consider-using-in,no-else-return
from dataclasses import dataclass
from typing import List, Optional
from src.domain.diagnosis.models import DiagnosticResult, ErrorType, ErrorCategory

@dataclass
class PedagogicalFeedback:
    summary: str
    actionable_tip: str
    detailed_explanation: str
    audio_example_ref: Optional[str] = None
    target_pinyin: str = ""
    student_pinyin: str = ""

class PedagogicalFeedbackGenerator:
    """
    Transforms technical diagnostic results into student-friendly pedagogical feedback.
    """

    @staticmethod
    def generate_feedback(diagnosis: DiagnosticResult) -> PedagogicalFeedback:
        err_type = diagnosis.error_type
        syl = diagnosis.syllable_text
        pinyin = diagnosis.pinyin

        if diagnosis.category == ErrorCategory.INSUFFICIENT_EVIDENCE:
            return PedagogicalFeedback(
                summary="Evidencia insuficiente para evaluar.",
                actionable_tip="Por favor, graba nuevamente tu voz en un entorno silencioso y pronunciando claramente.",
                detailed_explanation="La calidad del audio o la confianza del modelo no permiten emitir un diagnóstico preciso.",
                target_pinyin=pinyin,
                student_pinyin="?"
            )

        if err_type == ErrorType.CONFUSION_PAIR or err_type == ErrorType.RETROFLEXION_MISSING:
            target_ph = diagnosis.technical_details.get("target_phoneme", "")
            detected_ph = diagnosis.technical_details.get("detected_phoneme", "")

            if target_ph in ("zh", "ch", "sh") and detected_ph in ("z", "c", "s"):
                return PedagogicalFeedback(
                    summary=f"Falta de retroflexión en '{syl}' ({pinyin}).",
                    actionable_tip=f"Curva la punta de la lengua hacia atrás tocando el paladar duro para pronunciar '{target_ph}', evitando el sonido plano '{detected_ph}'.",
                    detailed_explanation=f"Has sustituido la consonante retrofleja '{target_ph}' por la consonante dental '{detected_ph}'.",
                    target_pinyin=pinyin,
                    student_pinyin=pinyin.replace(target_ph, detected_ph)
                )
            elif target_ph in ("z", "c", "s") and detected_ph in ("zh", "ch", "sh"):
                return PedagogicalFeedback(
                    summary=f"Retroflexión innecesaria en '{syl}' ({pinyin}).",
                    actionable_tip=f"Manten la lengua plana tras los dientes superiores para pronunciar '{target_ph}', sin curvarla hacia el paladar.",
                    detailed_explanation=f"Has sustituido la consonante dental '{target_ph}' por la retrofleja '{detected_ph}'.",
                    target_pinyin=pinyin,
                    student_pinyin=pinyin.replace(target_ph, detected_ph)
                )

        if err_type == ErrorType.INSUFFICIENT_FALL:
            return PedagogicalFeedback(
                summary=f"Caída tonal insuficiente en '{syl}' ({pinyin}).",
                actionable_tip="El tono 4 debe caer con fuerza y rapidez desde un registro alto hasta un registro bajo. Intenta hacer una caída más marcada.",
                detailed_explanation="Tu tono empezó correctamente en un registro alto, pero la pendiente de descenso fue demasiado suave.",
                target_pinyin=pinyin,
                student_pinyin=pinyin
            )

        if err_type == ErrorType.INSUFFICIENT_RISE:
            return PedagogicalFeedback(
                summary=f"Ascenso tonal insuficiente en '{syl}' ({pinyin}).",
                actionable_tip="El tono 2 debe subir claramente desde un registro medio hasta un registro alto, similar a una pregunta en español.",
                detailed_explanation="Tu trayectoria de tono no mostró la elevación necesaria en la segunda mitad de la sílaba.",
                target_pinyin=pinyin,
                student_pinyin=pinyin
            )

        if err_type == ErrorType.INSUFFICIENT_DIP:
            return PedagogicalFeedback(
                summary=f"Descenso-ascenso insuficiente para el Tono 3 en '{syl}' ({pinyin}).",
                actionable_tip="Baja primero la voz a tu registro más grave y luego permite que suba ligeramente.",
                detailed_explanation="El tono 3 requiere descender al fondo del registro vocal antes de recuperar altura.",
                target_pinyin=pinyin,
                student_pinyin=pinyin
            )

        if err_type == ErrorType.TONE_MISCLASSIFICATION:
            target_t = diagnosis.technical_details.get("target_tone", "")
            pred_t = diagnosis.technical_details.get("predicted_tone", "")
            return PedagogicalFeedback(
                summary=f"Tono incorrecto en '{syl}' ({pinyin}).",
                actionable_tip=f"Has producido el Tono {pred_t} en lugar del Tono {target_t}. Revisa la curva melódica requerida.",
                detailed_explanation=f"Se esperaba Tono {target_t} pero el contorno acústico se asemeja al Tono {pred_t}.",
                target_pinyin=pinyin,
                student_pinyin=pinyin
            )

        # Fallback general feedback
        return PedagogicalFeedback(
            summary=f"Revisa la pronunciación de '{syl}' ({pinyin}).",
            actionable_tip="Escucha atentamente el audio de referencia y repite prestando atención a la articulación y el contorno melódico.",
            detailed_explanation=f"Se detectó una discrepancia en {diagnosis.category.value} ({diagnosis.error_type.value}).",
            target_pinyin=pinyin,
            student_pinyin=pinyin
        )
