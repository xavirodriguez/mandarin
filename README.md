# Mandarin Computer-Assisted Pronunciation Training (CAPT) System

A Mandarin Chinese Computer-Assisted Pronunciation Training system built with Clean / Hexagonal Architecture.

## Setup & Installation

### Base Installation (without heavy ML models)
```bash
pip install -e .
```

### Development & ML Extensions
```bash
pip install -e ".[dev,ml]"
```

### UI Extension (Gradio)
```bash
pip install -e ".[ui]"
```

## Interfaz Gradio (UI)

El proyecto incluye una interfaz web interactiva construida con Gradio para grabar audio o subir archivos y evaluar la pronunciación en mandarín.

### 1. Instalación
Para instalar la interfaz junto con sus dependencias (`gradio` y `requests`):
```bash
pip install -e ".[ui]"
```

### 2. Ejecutar la API y la UI en paralelo
Abre dos terminales:

- **Terminal 1 (Servidor API):**
  ```bash
  uvicorn src.interfaces.api.routes:app --host 0.0.0.0 --port 8000 --reload
  ```

- **Terminal 2 (Interfaz Gradio):**
  ```bash
  python src/interfaces/ui/gradio_app.py
  ```

La interfaz estará disponible localmente en `http://localhost:7860`.

### 3. Apuntar la UI a una API desplegada (ej. Render)
Si el backend FastAPI se encuentra desplegado en un servidor remoto (por ejemplo, Render), puedes redirigir la UI estableciendo la variable de entorno `API_URL`:

```bash
export API_URL="https://tu-app.onrender.com/api/v1/assess"
python src/interfaces/ui/gradio_app.py
```

### 4. Formato de audio y timeouts
- **Formato esperado:** La UI convierte y remuestra automáticamente cualquier entrada de micrófono o archivo subido a audio **PCM int16 mono a 16 kHz**.
- **Timeout HTTP:** Las llamadas desde la UI tienen configurado un tiempo límite (timeout) de **120 segundos** para tolerar posibles *cold starts* (arrantes en frío) en instancias gratuitas de despliegue como Render.

## Running the API Server

Start the API server using `uvicorn`:
```bash
uvicorn src.interfaces.api.routes:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

- `CAPT_ADAPTER_MODE`: Set to `mock` (default) for fast unit testing with test doubles, or `real` for full neural model inference.
- `CAPT_MODEL_NAME`: Hugging Face model repository/name for acoustic recognition (default: `uai-org/wav2vec2-large-xlsr-53-mandarin-pinyin` or `TencentGameMate/chinese-wav2vec2-base`).

## API Endpoints

- `GET /health` - Service health status.
- `POST /api/v1/assess` - Pronunciation assessment endpoint evaluating phonetic accuracy, tone contours, and generating diagnostic feedback.

## Running Tests

Run unit and evaluation tests:
```bash
pytest
```
