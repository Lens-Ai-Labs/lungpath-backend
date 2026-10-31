import io
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("DATA_DIR", "/tmp/lungpath_test_data")

from main import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_ready" in payload
    assert "model_loaded" in payload


def test_analyses_list_endpoint():
    response = client.get("/api/analyses")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_reject_invalid_file_type():
    response = client.post(
        "/api/analyze",
        files={"file": ("notes.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 400
