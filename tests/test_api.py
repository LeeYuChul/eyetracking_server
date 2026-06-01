import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def make_png() -> bytes:
    image = Image.new("RGBA", (80, 120), (255, 255, 255, 255))
    for x in range(30, 55):
        for y in range(40, 80):
            image.putpixel((x, y), (220, 20, 20, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["port"] == 3781
    assert response.json()["model_loaded"] is True
    assert response.json()["models"]["umsi++"]["weights_path"].endswith("umsi++.hdf5")


def test_analysis_returns_report_without_storage_urls():
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("frame.png", make_png(), "image/png")},
        data={"frame_id": "1:2", "frame_name": "Home", "width": "80", "height": "120", "model_name": "umsi++"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["report"]["model"]["name"] == "UMSI++"
    assert body["assets"]["image_mime_type"] == "image/png"
    assert body["assets"]["heatmap_png_base64"]
    assert body["assets"]["overlay_png_base64"]
    assert "heatmap_url" not in body


def test_analysis_accepts_heuristic_model_selection():
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("frame.png", make_png(), "image/png")},
        data={"frame_id": "1:2", "frame_name": "Home", "width": "80", "height": "120", "model_name": "heuristic"},
    )
    assert response.status_code == 200
    assert response.json()["report"]["model"]["name"] == "Heuristic"


def test_openapi_only_documents_live_apis():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert sorted(paths) == ["/api/v1/analyses", "/api/v1/health"]


def test_invalid_file_type():
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("frame.txt", b"not image", "text/plain")},
        data={"frame_id": "1:2", "frame_name": "Home", "width": "80", "height": "120"},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_FILE_TYPE"
