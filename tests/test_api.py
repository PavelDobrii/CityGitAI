import os
import types
import tempfile
from fastapi.testclient import TestClient
import importlib
import subprocess

# Import the app after setting up environment
from app import main as app_main


def dummy_translate(text, src='en', dest='ru'):
    return types.SimpleNamespace(text=text)


def dummy_run(*args, **kwargs):
    class Dummy:
        stdout = b""
        stderr = b""
    return Dummy()

def failing_run(*args, **kwargs):
    raise subprocess.CalledProcessError(1, args[0], output=b"", stderr=b"boom")


def setup_test_environment(tmp_dir, monkeypatch):
    app_main.OUTPUT_DIR = tmp_dir
    os.makedirs(app_main.OUTPUT_DIR, exist_ok=True)
    app_main.DEFAULT_LANG = "en"
    app_main.DEFAULT_VOICE_EN = "voice-en.onnx"
    app_main.DEFAULT_VOICE_RU = "voice-ru.onnx"
    monkeypatch.setattr(app_main, "ollama_generate", lambda *a, **k: "story")
    monkeypatch.setattr(app_main, "get_wikipedia_summary", lambda *a, **k: "wiki")
    monkeypatch.setattr(app_main, "get_wikivoyage_intro", lambda *a, **k: "voyage")
    monkeypatch.setattr(app_main, "get_osm_description", lambda *a, **k: "osm")
    monkeypatch.setattr(app_main.translator, "translate", dummy_translate)
    monkeypatch.setattr(app_main.subprocess, "run", dummy_run)


def test_generate_returns_paths(tmp_path, monkeypatch):
    setup_test_environment(str(tmp_path), monkeypatch)
    client = TestClient(app_main.app)
    payload = {"topic": "Test", "style": "style", "lang": "en"}
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "story_id" in data
    assert data["story_id"]
    assert data["markdown_path"].startswith(str(tmp_path))
    assert data["audio_path"].startswith(str(tmp_path))


def test_generate_handles_tts_error(tmp_path, monkeypatch):
    setup_test_environment(str(tmp_path), monkeypatch)
    monkeypatch.setattr(app_main.subprocess, "run", failing_run)
    client = TestClient(app_main.app)
    payload = {"topic": "Test", "style": "style"}
    resp = client.post("/generate", json=payload)
    assert resp.status_code == 200
    assert resp.json().get("error") == "TTS failed"

