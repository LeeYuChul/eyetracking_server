import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, model_registry
from app.models.frame import FrameChatRequest, FrameImageInput
from app.services.vlm import build_prompt


client = TestClient(app)


def make_png() -> bytes:
    image = Image.new("RGBA", (80, 120), (255, 255, 255, 255))
    for x in range(30, 55):
        for y in range(40, 80):
            image.putpixel((x, y), (220, 20, 20, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_png_base64() -> str:
    import base64

    return base64.b64encode(make_png()).decode("ascii")


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["port"] == 3781
    assert body["storage_policy"] == "stateless_response_only"
    assert body["model_loaded"] is False
    assert body["model_status"]["heatmap_model_loaded"] is False
    assert body["model_status"]["scanpath_model_loaded"] is True
    assert body["model_status"]["vlm_provider"] == "ollama"


def test_openapi_exposes_frame_chat_contract_without_flow_or_heuristic_chat():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/health" in paths
    assert "/api/v1/analyses" in paths
    assert "/api/v1/frames/analyze" in paths
    assert "/api/v1/frames/chat/stream" in paths
    assert not any(path.startswith("/api/v1/flow/") for path in paths)
    assert "/api/v1/ux/chat/heuristic" not in paths
    assert "/api/v1/ux/chat" not in paths
    assert "/api/v1/ux/evaluate" not in paths


def test_frames_analyze_returns_individual_frame_results():
    response = client.post(
        "/api/v1/frames/analyze",
        files=[
            ("files", ("frame_a", make_png(), "image/png")),
            ("files", ("frame_b", make_png(), "image/png")),
        ],
        data={
            "model_name": "heuristic",
            "frames_meta": json.dumps(
                [
                    {
                        "client_frame_id": "local_1",
                        "figma_node_id": "1:1",
                        "frame_name": "Home",
                        "width": 80,
                        "height": 120,
                        "file_key": "frame_a",
                        "order_index": 0,
                    },
                    {
                        "client_frame_id": "local_2",
                        "figma_node_id": "1:2",
                        "frame_name": "Search",
                        "width": 80,
                        "height": 120,
                        "file_key": "frame_b",
                        "order_index": 1,
                    },
                ]
            ),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["storage_policy"] == "client_must_store_response"
    assert "flow_tree" not in body
    assert len(body["frames"]) == 2
    frame = body["frames"][0]
    assert frame["frame_name"] == "Home"
    assert frame["metrics"]["fixation_count"] > 0
    assert frame["artifacts"]["original"]["base64"]
    assert frame["artifacts"]["heatmap_overlay"]["base64"]
    assert frame["artifacts"]["scanpath_overlay"]["base64"]


def test_frames_analyze_rejects_too_many_frames():
    response = client.post(
        "/api/v1/frames/analyze",
        files=[("files", (f"frame_{index}", make_png(), "image/png")) for index in range(16)],
        data={
            "model_name": "heuristic",
            "frames_meta": json.dumps(
                [
                    {
                        "client_frame_id": f"local_{index}",
                        "frame_name": f"Screen{index}",
                        "width": 80,
                        "height": 120,
                        "file_key": f"frame_{index}",
                        "order_index": index,
                    }
                    for index in range(16)
                ]
            ),
        },
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_FRAME_COUNT"


def test_frames_analyze_rejects_metadata_file_mismatch():
    response = client.post(
        "/api/v1/frames/analyze",
        files=[("files", ("frame_a", make_png(), "image/png"))],
        data={
            "model_name": "heuristic",
            "frames_meta": json.dumps(
                [
                    {
                        "client_frame_id": "local_1",
                        "frame_name": "Home",
                        "width": 80,
                        "height": 120,
                        "file_key": "missing",
                        "order_index": 0,
                    }
                ]
            ),
        },
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_FRAME_METADATA"


def test_frame_chat_stream_returns_progress_and_final(monkeypatch):
    async def fake_stream_events(request, settings):
        assert len(request.selected_images) == 3
        yield {"event": "progress", "data": {"stage": "started", "message": "시작", "progress": 0}}
        yield {"event": "thinking_delta", "data": {"delta": "CTA 위치를 확인합니다.", "progress": 0.45}}
        yield {"event": "answer_delta", "data": {"delta": "CTA 주변 시선 집중", "progress": 0.7}}
        yield {
            "event": "final",
            "data": {
                "answer": {
                    "conclusion": "CTA 주변 시선 집중은 중간 수준입니다.",
                    "reasoning_summary": ["Heatmap과 Scanpath 이미지를 확인했습니다."],
                    "evidence_images": ["original", "heatmap_overlay", "scanpath_overlay"],
                    "risk_level": "medium",
                    "recommendations": ["CTA 주변 대비를 높이세요."],
                    "confidence": "medium",
                    "caveat": "predictive only",
                },
                "provider": "ollama",
                "model": "gemma4:26b",
                "progress": 1,
            },
        }

    monkeypatch.setattr("app.main.stream_frame_chat_events", fake_stream_events)
    artifact = {
        "artifact_type": "original",
        "mime_type": "image/png",
        "base64": make_png_base64(),
        "width": 80,
        "height": 120,
    }
    response = client.post(
        "/api/v1/frames/chat/stream",
        json={
            "question": "CTA가 잘 보일까?",
            "frame_id": "local_1",
            "frame_name": "Home",
            "metrics": {"scanpath_length": 300, "fixation_count": 4},
            "selected_images": [
                {**artifact, "artifact_type": "original", "image_role": "original"},
                {**artifact, "artifact_type": "heatmap_overlay", "image_role": "heatmap_overlay"},
                {**artifact, "artifact_type": "scanpath_overlay", "image_role": "scanpath_overlay"},
            ],
            "previous_messages": [],
        },
    )
    assert response.status_code == 200
    assert "event: progress" in response.text
    assert "event: thinking_delta" in response.text
    assert "event: answer_delta" in response.text
    assert "event: final" in response.text
    assert "CTA 주변 시선 집중" in response.text


def test_frame_chat_prompt_prioritizes_user_question():
    artifact = FrameImageInput(
        artifact_type="original",
        mime_type="image/png",
        base64=make_png_base64(),
        width=80,
        height=120,
        image_role="original",
    )
    request = FrameChatRequest(
        question="스타터는 어떤 의미일까?",
        frame_id="local_1",
        frame_name="마이페이지",
        metrics={"fixation_count": 8},
        selected_images=[artifact],
        previous_messages=[],
    )
    prompt = build_prompt(request, streaming=True)
    assert "Question에 직접 답" in prompt
    assert "용어 의미" in prompt
    assert "마지막에는 사용자가 바로 수정할 수 있는 제안" not in prompt
    assert "사용자가 묻지 않은 주제" in prompt


def test_legacy_single_analysis_still_works():
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("frame.png", make_png(), "image/png")},
        data={"frame_id": "1:2", "frame_name": "Home", "width": "80", "height": "120", "model_name": "heuristic"},
    )
    assert response.status_code == 200
    assert response.json()["report"]["model"]["name"] == "Heuristic"
    assert model_registry.get("heuristic").loaded is False
