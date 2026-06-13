import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, model_registry
from app.models.flow import UxAnswer, UxEvaluateResponse


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
    assert body["models"]["umsi++"]["weights_path"].endswith("umsi++.hdf5")


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
    assert model_registry.get("heuristic").loaded is True


def test_openapi_documents_flow_apis_and_legacy_analysis():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert sorted(paths) == [
        "/api/v1/analyses",
        "/api/v1/flow/analyze",
        "/api/v1/flow/parse",
        "/api/v1/flow/prepare-target",
        "/api/v1/health",
        "/api/v1/ux/chat",
        "/api/v1/ux/chat/heuristic",
        "/api/v1/ux/evaluate",
    ]


def test_invalid_file_type():
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("frame.txt", b"not image", "text/plain")},
        data={"frame_id": "1:2", "frame_name": "Home", "width": "80", "height": "120"},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_FILE_TYPE"


def test_flow_parse_warns_for_state_and_invalid_names():
    response = client.post(
        "/api/v1/flow/parse",
        json={
            "frames": [
                {"client_frame_id": "local_1", "frame_name": "1_1_Home", "order_index": 0},
                {"client_frame_id": "local_2", "frame_name": "1_1-1_Home-Expanded", "order_index": 1},
                {"client_frame_id": "local_3", "frame_name": "Bad Name", "order_index": 2},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["parsed_frames"][0]["flow_id"] == "1"
    assert body["parsed_frames"][1]["state"] == 1
    assert body["parsed_frames"][2]["parse_status"] == "unparsed"
    assert body["flow_tree"]["flows"][0]["ordered_frame_ids"] == ["local_1", "local_2"]
    assert body["warnings"][0]["code"] == "UNPARSED_FRAME_NAME"


def test_flow_analyze_returns_bundle_with_scanpath_artifacts():
    response = client.post(
        "/api/v1/flow/analyze",
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
                        "frame_name": "1_1_Home",
                        "width": 80,
                        "height": 120,
                        "file_key": "frame_a",
                        "order_index": 0,
                    },
                    {
                        "client_frame_id": "local_2",
                        "figma_node_id": "1:2",
                        "frame_name": "1_2_Search",
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
    assert body["flow_tree"]["flows"][0]["ordered_frame_ids"] == ["local_1", "local_2"]
    assert len(body["frames"]) == 2
    frame = body["frames"][0]
    assert frame["metrics"]["fixation_count"] > 0
    assert frame["metrics"]["scanpath_length"] >= 0
    assert frame["artifacts"]["heatmap"]["base64"]
    assert frame["artifacts"]["scanpath_overlay"]["base64"]
    assert frame["artifacts"]["heatmap_overlay"]["base64"]
    assert frame["artifacts"]["original"]["base64"]


def test_flow_analyze_rejects_too_many_frames():
    response = client.post(
        "/api/v1/flow/analyze",
        files=[("files", (f"frame_{index}", make_png(), "image/png")) for index in range(16)],
        data={
            "model_name": "heuristic",
            "frames_meta": json.dumps(
                [
                    {
                        "client_frame_id": f"local_{index}",
                        "frame_name": f"1_{index + 1}_Screen{index}",
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


def test_prepare_target_returns_memory_blur_for_prior_path_frames():
    analyze_response = client.post(
        "/api/v1/flow/analyze",
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
                        "frame_name": "1_1_Home",
                        "width": 80,
                        "height": 120,
                        "file_key": "frame_a",
                        "order_index": 0,
                    },
                    {
                        "client_frame_id": "local_2",
                        "frame_name": "1_2_Search",
                        "width": 80,
                        "height": 120,
                        "file_key": "frame_b",
                        "order_index": 1,
                    },
                ]
            ),
        },
    )
    bundle = analyze_response.json()
    response = client.post(
        "/api/v1/flow/prepare-target",
        json={
            "target_frame_id": "local_2",
            "flow_tree": bundle["flow_tree"],
            "frames": [
                {
                    "client_frame_id": frame["client_frame_id"],
                    "frame_name": frame["frame_name"],
                    "original_image": frame["artifacts"]["original"],
                    "heatmap": frame["artifacts"]["heatmap"],
                    "scanpath_metrics": frame["metrics"],
                }
                for frame in bundle["frames"]
            ],
            "options": {"blur_strength_base": 4},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["target_frame_id"] == "local_2"
    assert body["path_frame_ids"] == ["local_1", "local_2"]
    assert body["frames"][0]["temporal_distance"] == 1
    assert body["frames"][0]["artifacts"]["memory_blur"]["base64"]
    assert body["frames"][0]["artifacts"]["full_overlay"]["base64"]


def test_prepare_target_returns_memory_for_full_15_frame_path():
    artifact = {
        "artifact_type": "original",
        "mime_type": "image/png",
        "base64": make_png_base64(),
        "width": 80,
        "height": 120,
    }
    response = client.post(
        "/api/v1/flow/prepare-target",
        json={
            "target_frame_id": "local_15",
            "flow_tree": {
                "flows": [
                    {
                        "flow_id": "1",
                        "frames": [],
                        "ordered_frame_ids": [f"local_{index}" for index in range(1, 16)],
                    }
                ],
                "unparsed_frame_ids": [],
                "ordered_frame_ids": [f"local_{index}" for index in range(1, 16)],
            },
            "frames": [
                {
                    "client_frame_id": f"local_{index}",
                    "frame_name": f"1_{index}_Screen",
                    "original_image": artifact,
                    "heatmap": artifact,
                    "scanpath_metrics": {"scanpath_length": 100 + index},
                }
                for index in range(1, 16)
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["path_frame_ids"] == [f"local_{index}" for index in range(1, 16)]
    assert len(body["frames"]) == 14
    assert body["frames"][0]["temporal_distance"] == 14
    assert body["frames"][-1]["temporal_distance"] == 1


def test_prepare_target_rejects_missing_target():
    response = client.post(
        "/api/v1/flow/prepare-target",
        json={
            "target_frame_id": "missing",
            "flow_tree": {"flows": [], "unparsed_frame_ids": [], "ordered_frame_ids": []},
            "frames": [],
        },
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "TARGET_PATH_NOT_FOUND"


def test_ux_evaluate_uses_configured_provider(monkeypatch):
    async def fake_evaluate(request, settings):
        return UxEvaluateResponse(
            answer=UxAnswer(
                conclusion="기억 가능성은 중간입니다.",
                reasoning_summary=["target path와 memory metric을 확인했습니다."],
                evidence_frames=["local_1"],
                risk_level="medium",
                recommendations=["목표 화면에 확인 단서를 추가하세요."],
                confidence="medium",
                caveat="predictive only",
            ),
            provider=settings.vlm_provider,
            model=settings.ollama_model,
        )

    monkeypatch.setattr("app.main.evaluate_ux", fake_evaluate)
    response = client.post(
        "/api/v1/ux/evaluate",
        json={
            "question": "사용자가 닉네임을 기억할까?",
            "target_frame_id": "local_2",
            "flow_tree": {"flows": [], "unparsed_frame_ids": [], "ordered_frame_ids": []},
            "evidence": {"frames": []},
            "previous_messages": [],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "ollama"
    assert body["answer"]["recommendations"]


def test_ux_chat_uses_configured_provider(monkeypatch):
    async def fake_evaluate(request, settings):
        return UxEvaluateResponse(
            answer=UxAnswer(
                conclusion="챗봇 응답입니다.",
                reasoning_summary=["VLM chat endpoint를 사용했습니다."],
                evidence_frames=["local_2"],
                risk_level="low",
                recommendations=["현재 흐름을 유지하세요."],
                confidence="medium",
                caveat="predictive only",
            ),
            provider=settings.vlm_provider,
            model=settings.ollama_model,
        )

    monkeypatch.setattr("app.main.evaluate_ux", fake_evaluate)
    response = client.post(
        "/api/v1/ux/chat",
        json={
            "question": "이 플로우에서 CTA를 찾을 수 있을까?",
            "target_frame_id": "local_2",
            "flow_tree": {"flows": [], "unparsed_frame_ids": [], "ordered_frame_ids": []},
            "evidence": {
                "frames": [],
                "selected_images": [
                    {
                        "artifact_type": "original",
                        "mime_type": "image/png",
                        "base64": make_png_base64(),
                        "width": 80,
                        "height": 120,
                    }
                ],
            },
            "previous_messages": [{"role": "user", "content": "이전 질문"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "ollama"
    assert body["answer"]["conclusion"] == "챗봇 응답입니다."


def test_ux_chat_sends_target_flow_images_to_ollama(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "conclusion": "목표 화면의 공고 분석 버튼을 선택합니다.",
                            "reasoning_summary": ["target frame image를 직접 확인했습니다."],
                            "evidence_frames": ["local_15"],
                            "risk_level": "medium",
                            "recommendations": ["CTA 라벨을 더 명확하게 유지하세요."],
                            "confidence": "medium",
                            "caveat": "predictive only",
                        },
                        ensure_ascii=False,
                    )
                }
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            captured["url"] = url
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr("app.services.vlm.httpx.AsyncClient", FakeAsyncClient)
    selected_images = [
        {
            "artifact_type": "original",
            "mime_type": "image/png",
            "base64": make_png_base64(),
            "width": 80,
            "height": 120,
            "frame_id": f"local_{index}",
            "frame_name": f"1_{index}_Screen",
            "image_role": "flow_original",
        }
        for index in range(1, 16)
    ]
    selected_images.extend(
        [
            {
                "artifact_type": "full_overlay",
                "mime_type": "image/png",
                "base64": make_png_base64(),
                "width": 80,
                "height": 120,
                "frame_id": "local_15",
                "frame_name": "1_15_Target",
                "image_role": "target_overlay",
            }
            for _ in range(5)
        ]
    )
    response = client.post(
        "/api/v1/ux/chat",
        json={
            "question": "AI로 공고 분석하기를 들어가려면 어떤 버튼을 누르면 될까?",
            "target_frame_id": "local_15",
            "flow_tree": {"flows": [], "unparsed_frame_ids": [], "ordered_frame_ids": [f"local_{index}" for index in range(1, 16)]},
            "evidence": {
                "frames": [
                    {
                        "client_frame_id": f"local_{index}",
                        "frame_name": f"1_{index}_Screen",
                        "metrics": {"scanpath_length": 300, "visual_complexity": 0.3},
                    }
                    for index in range(1, 16)
                ],
                "target_result": {
                    "target_frame_id": "local_15",
                    "path_frame_ids": [f"local_{index}" for index in range(1, 16)],
                    "frames": [],
                },
                "selected_images": selected_images,
            },
            "previous_messages": [],
        },
    )
    assert response.status_code == 200
    payload = captured["payload"]
    assert payload["model"] == "gemma4:26b"
    assert len(payload["messages"][1]["images"]) == 20
    assert "Target frame: local_15" in payload["messages"][1]["content"]
    assert "Do not answer only with memory-retention metrics" in payload["messages"][1]["content"]


def test_ux_chat_requires_selected_images():
    response = client.post(
        "/api/v1/ux/chat",
        json={
            "question": "어떤 버튼을 눌러야 해?",
            "target_frame_id": "local_2",
            "flow_tree": {"flows": [], "unparsed_frame_ids": [], "ordered_frame_ids": []},
            "evidence": {"frames": []},
            "previous_messages": [],
        },
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_FRAME_METADATA"


def test_ux_heuristic_chat_returns_metric_based_answer():
    response = client.post(
        "/api/v1/ux/chat/heuristic",
        json={
            "question": "사용자가 첫 화면의 정보를 기억할까?",
            "target_frame_id": "local_3",
            "flow_tree": {
                "flows": [],
                "unparsed_frame_ids": [],
                "ordered_frame_ids": ["local_1", "local_2", "local_3"],
            },
            "evidence": {
                "frames": [
                    {
                        "client_frame_id": "local_1",
                        "frame_name": "1_1_Home",
                        "metrics": {
                            "scanpath_length": 1400,
                            "visual_complexity": 0.8,
                        },
                    }
                ],
                "target_result": {
                    "target_frame_id": "local_3",
                    "path_frame_ids": ["local_1", "local_2", "local_3"],
                    "frames": [
                        {
                            "client_frame_id": "local_1",
                            "temporal_distance": 2,
                            "memory_metrics": {
                                "estimated_retention": 0.22,
                                "temporal_distance": 2,
                            },
                        }
                    ],
                },
            },
            "previous_messages": [],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "heuristic"
    assert body["model"] == "ux-flow-heuristic-chat-0.1"
    assert body["answer"]["risk_level"] == "high"
    assert body["answer"]["recommendations"]
