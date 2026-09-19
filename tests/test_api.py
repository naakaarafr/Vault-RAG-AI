from fastapi.testclient import TestClient

from vault.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "vault-api"

def test_root_frontend_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Vault AI Workspace" in response.text

def test_ask_endpoint():
    payload = {"question": "What is the system architecture of Vault?", "role": "admin"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "refused" in data
    assert "latency_ms" in data
