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
