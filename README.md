# Mandarin CAPT (Computer-Assisted Pronunciation Training)

Un sistema de evaluación de pronunciación y tono en chino mandarín basado en una arquitectura en capas (`domain` → `application` → `infrastructure` → `interfaces`).

## Características Principales

- **Evaluación Fonética y GOP**: Puntuación Goodness of Pronunciation (GOP) utilizando modelos acústicos con alineación temporal.
- **Clasificación y Evaluación Tonal**: Análisis de contorno F0 normalizado por hablante y reglas fonológicas de sandhi tonal.
- **Calibración de Scores**: Módulo `ModelCalibrator` con escalado de temperatura para una evaluación justa e incertidumbre calibrada.
- **Diagnóstico y Feedback Pedagógico**: Identificación de pares de confusión (ej. retroflejas vs dentales) y generación de consejos de articulación.
- **Toggle Adaptadores Mock / Real**: Posibilidad de alternar entre pruebas ligeras con mocks y el pipeline completo con modelos neuronales (Wav2Vec2 / XLS-R).

---

## Instalación

1. Clonar el repositorio y navegar a la raíz:
```bash
git clone https://github.com/xavirodriguez/mandarin.git
cd mandarin
```

2. Instalar dependencias de desarrollo (modo rápido con mocks):
```bash
pip install -e ".[dev]"
```

3. (Opcional) Instalar dependencias de Machine Learning (Torch, Transformers, HuggingFace):
```bash
pip install -e ".[dev,ml]"
```

---

## Variables de Entorno

| Variable | Descripción | Valores Posibles | Valor por Defecto |
|---|---|---|---|
| `CAPT_ADAPTER_MODE` | Determina si se ejecutan componentes mock o modelos reales | `mock`, `real` | `mock` |
| `CAPT_MODEL_NAME` | Modelo preentrenado de HuggingFace para codificador y reconocimiento | Nombre del repositorio HF | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` |

---

## Uso de la API (HTTP/FastAPI)

Iniciar el servidor API local con Uvicorn:
```bash
uvicorn src.interfaces.api.routes:app --reload --port 8000
```

### Endpoints Principales

- **`GET /health`**: Verificación del estado del servicio.
- **`POST /api/v1/assess`**: Evaluación de audio comprimido en Base64 (WAV/PCM).
- **`POST /api/v1/assess/upload`**: Evaluación vía subida directa de archivos WAV (`UploadFile`).

Ejemplo de payload para `POST /api/v1/assess`:
```json
{
  "target_pinyin": ["ní", "hǎo"],
  "lexical_tones": [3, 3],
  "audio_base64": "<base64_wav_data>"
}
```

---

## Ejecución de Tests y Benchmark

### Correr Pruebas Unitarias e Integración
```bash
python3 -m pytest
```

### Ejecutar Pylint
```bash
pylint $(git ls-files '*.py')
```

### Ejecutar Benchmark Cuantitativo
Para evaluar la precisión fonética, precisión tonal y diagnósticos en splits *speaker-independent*:
```bash
python3 -m src.evaluation.benchmark
```

---

## Documentación Relevante

- **[SPIKE_MODEL.md](SPIKE_MODEL.md)**: Informe de evaluación del modelo acústico Wav2Vec2/XLS-R.
- **[BASELINE.md](BASELINE.md)**: Resultados cuantitativos del benchmark (Tone Accuracy, GOP Correlation, Diagnostic F1).
