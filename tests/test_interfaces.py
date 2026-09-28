from fastapi.testclient import TestClient
from src.interfaces.api.routes import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "Mandarin CAPT System"}

def test_api_assess_endpoint():
    payload = {
        "target_pinyin": ["yī", "yàng"],
        "lexical_tones": [1, 4]
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
