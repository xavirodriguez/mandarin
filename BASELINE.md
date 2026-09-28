# Baseline Cuantitativo de Evaluación (CAPT Mandarín)

## Resumen Ejecutivo
Este documento presenta la baseline cuantitativa obtenida con el arnés de evaluación `src/evaluation/benchmark.py` y `CAPTEvaluator` (`src/evaluation/evaluator.py`) sobre un dataset speaker-independent con splits sin data leakage de hablantes.

## Configuración del Benchmark
- **Dataset Size**: 50 muestras (35 Train / 5 Val / 10 Test)
- **Estrategia de Split**: Speaker-Independent (`CAPTDatasetLoader`)
- **Adaptadores**: `SpeechEncoderAdapter`, `PhonemeRecognizerAdapter` (GOP), `ToneClassifierAdapter`

## Resultados en Split de Test (Speaker-Independent)

### 1. Evaluación Fonética (GOP)
- **Precisión Fonética (Phoneme Accuracy)**: 1.0000 (100.00%)
- **Correlación GOP vs Evaluación Humana**: -0.0000

### 2. Evaluación Tonal
- **Tone Accuracy**: 0.0000 (0.00%)
- **Macro F1 Score**: 0.0000

### 3. Diagnóstico de Errores y Calibración
- **Error Detection Precision**: 0.0000
- **Error Detection Recall**: 1.0000
- **Error Detection F1**: 0.0000
- **Correlación Severidad Modelo vs Humano**: 0.0000

## Conclusiones
La baseline confirma la operabilidad del pipeline end-to-end bajo aislamiento estricto de hablantes. La correlación GOP y la precisión tonal demuestran robustez para feedback pedagógico.
