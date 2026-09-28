import io
import base64
import soundfile as sf
import numpy as np
from fastapi.testclient import TestClient
from src.interfaces.api.routes import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "Mandarin CAPT System"}

def create_dummy_wav_base64(duration_sec: float = 1.0, sr: int = 16000) -> str:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 300 * t)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format='WAV')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def test_api_assess_missing_audio():
    # Should return HTTP 400 when audio_base64 is missing
    payload = {
        "target_pinyin": ["yī", "yàng"],
        "lexical_tones": [1, 4]
    }
    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 400
    assert "Audio is required" in response.json()["detail"]

def test_api_assess_endpoint_valid_wav():
    audio_b64 = create_dummy_wav_base64(duration_sec=1.0)
    payload = {
        "target_pinyin": ["yī", "yàng"],
        "lexical_tones": [1, 4],
        "audio_base64": audio_b64
    }
    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["utterance"] == "yī yàng"
    assert "audio_quality" in data
    assert "syllables" in data
    assert "errors" in data
    assert "feedback" in data

    # Verify Sandhi in API output
    assert len(data["syllables"]) == 2
    syl0 = data["syllables"][0]
    assert syl0["tone"]["contextual_target"] == 2
    assert syl0["tone"]["is_sandhi_applied"] is True

def test_api_assess_endpoint_with_user_id_persistence():
    payload = {
        "user_id": "student_test_42",
        "target_pinyin": ["ní", "hǎo"],
        "lexical_tones": [3, 3]
    }
    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["assessment_id"] is not None
    assert isinstance(data["assessment_id"], str)
