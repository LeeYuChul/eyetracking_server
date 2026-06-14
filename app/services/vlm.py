from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from app.core.config import Settings
from app.models.frame import FrameChatAnswer, FrameChatRequest, FrameChatResponse
from app.services.artifacts import artifact_to_data_url, image_from_artifact, image_to_base64

CHAT_CAVEAT = "이 답변은 아이트래킹 예측 모델의 heatmap/scanpath 결과 기반 참고이며 실제 사용자 테스트를 대체하지 않습니다."
MAX_FRAME_CHAT_IMAGES = 3


async def evaluate_frame_chat(request: FrameChatRequest, settings: Settings) -> FrameChatResponse:
    provider = settings.vlm_provider.strip().lower()
    if provider == "openai":
        answer = await evaluate_frame_with_openai(request, settings)
        return FrameChatResponse(answer=answer, provider="openai", model=settings.openai_model)
    if provider == "ollama":
        answer = await evaluate_frame_with_ollama(request, settings)
        return FrameChatResponse(answer=answer, provider="ollama", model=settings.ollama_model)
    raise RuntimeError(f"Unsupported VLM provider: {settings.vlm_provider}")


async def evaluate_frame_with_ollama(request: FrameChatRequest, settings: Settings) -> FrameChatAnswer:
    if not request.selected_images:
        raise RuntimeError("Frame chat requires selected_images")
    async with httpx.AsyncClient(timeout=settings.vlm_request_timeout_seconds) as client:
        payload = build_ollama_payload(settings=settings, messages=build_ollama_messages(request, settings))
        content = await post_ollama_chat(client, settings, payload, image_count=len(request.selected_images[:MAX_FRAME_CHAT_IMAGES]))
    return parse_answer(content)


async def stream_frame_chat_events(request: FrameChatRequest, settings: Settings):
    if not request.selected_images:
        raise RuntimeError("Frame chat requires selected_images")

    yield stream_event(
        "progress",
        {
            "stage": "started",
            "message": "선택한 프레임의 원본, Heatmap, Scanpath 이미지를 확인합니다.",
            "progress": 0,
        },
    )
    async with httpx.AsyncClient(timeout=settings.vlm_request_timeout_seconds) as client:
        yield stream_event(
            "thinking",
            {
                "stage": "image_evidence",
                "message": "VLM에 원본/Heatmap/Scanpath 근거 이미지를 전달했습니다.",
                "evidence_images": image_roles(request),
                "progress": 0.35,
            },
        )
        payload = build_ollama_payload(settings=settings, messages=build_ollama_messages(request, settings))
        content = await post_ollama_chat(client, settings, payload, image_count=len(request.selected_images[:MAX_FRAME_CHAT_IMAGES]))
        answer = parse_answer(content)

    yield stream_event("progress", {"stage": "synthesizing", "message": "시선 지표와 이미지 근거를 종합 중입니다.", "progress": 0.85})
    yield stream_event(
        "final",
        {
            "answer": answer.model_dump(),
            "provider": "ollama",
            "model": settings.ollama_model,
            "progress": 1,
        },
    )


async def evaluate_frame_with_openai(request: FrameChatRequest, settings: Settings) -> FrameChatAnswer:
    if not settings.openai_api_key:
        raise RuntimeError("OpenAI API key is not configured")
    content: list[dict[str, Any]] = [{"type": "input_text", "text": build_prompt(request)}]
    for image in request.selected_images[:MAX_FRAME_CHAT_IMAGES]:
        content.append({"type": "input_image", "image_url": artifact_to_data_url(image)})
    payload = {
        "model": settings.openai_model,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "frame_chat_answer",
                "schema": answer_schema(),
                "strict": True,
            }
        },
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=settings.vlm_request_timeout_seconds) as client:
        response = await client.post(f"{settings.openai_base_url.rstrip('/')}/responses", json=payload, headers=headers)
        response.raise_for_status()
    return parse_answer(response.json().get("output_text", ""))


def build_ollama_payload(*, settings: Settings, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": settings.ollama_model,
        "stream": False,
        "messages": messages,
        "format": "json",
        "think": False,
        "keep_alive": settings.ollama_keep_alive,
        "options": {
            "num_ctx": settings.ollama_num_ctx,
            "num_predict": settings.ollama_num_predict,
            "temperature": 0.1,
        },
    }


def build_ollama_messages(request: FrameChatRequest, settings: Settings) -> list[dict[str, Any]]:
    message: dict[str, Any] = {"role": "user", "content": build_prompt(request)}
    images = request.selected_images[:MAX_FRAME_CHAT_IMAGES]
    message["images"] = [optimize_image_for_ollama(image, settings) for image in images]
    return [{"role": "system", "content": system_prompt()}, message]


async def post_ollama_chat(client: httpx.AsyncClient, settings: Settings, payload: dict[str, Any], *, image_count: int) -> str:
    response = await client.post(f"{settings.ollama_base_url.rstrip('/')}/api/chat", json=payload)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Ollama rejected frame chat request status={response.status_code} "
            f"image_count={image_count} message={safe_error_text(response.text)}"
        )
    response.raise_for_status()
    return str(response.json().get("message", {}).get("content", ""))


def build_prompt(request: FrameChatRequest) -> str:
    previous = [{"role": item.role, "content": item.content[:500]} for item in request.previous_messages[-6:]]
    return (
        f"Question: {request.question}\n"
        f"Frame ID: {request.frame_id}\n"
        f"Frame name: {request.frame_name or request.frame_id}\n"
        f"Metrics JSON: {json.dumps(request.metrics, ensure_ascii=False)}\n"
        f"Image roles JSON: {json.dumps(image_roles(request), ensure_ascii=False)}\n"
        f"Previous messages JSON: {json.dumps(previous, ensure_ascii=False)}\n\n"
        "Use the attached original screen, heatmap overlay, and scanpath overlay as primary evidence. "
        "Explain what the heatmap suggests users notice, where the scanpath likely moves, and how that affects the user's question. "
        "Do not discuss IA Flow, target paths, memory blur, or heuristic flow evaluation. "
        "Return only JSON with conclusion, reasoning_summary, evidence_images, risk_level, recommendations, confidence, caveat."
    )


def system_prompt() -> str:
    return (
        "You are an image-based UX reviewer focused on one Figma frame at a time. "
        "You answer from original UI, predicted eye-tracking heatmap, and scanpath evidence. "
        "Keep reasoning concise, practical, and grounded in visible UI and model outputs."
    )


def image_roles(request: FrameChatRequest) -> list[dict[str, Any]]:
    return [
        {
            "artifact_type": image.artifact_type,
            "image_role": image.image_role,
            "frame_id": image.frame_id or request.frame_id,
            "frame_name": image.frame_name or request.frame_name,
            "width": image.width,
            "height": image.height,
        }
        for image in request.selected_images[:MAX_FRAME_CHAT_IMAGES]
    ]


def optimize_image_for_ollama(image, settings: Settings) -> str:
    try:
        source = image_from_artifact(image).convert("RGB")
    except Exception:
        return image.base64.split(",", 1)[-1]

    max_side = max(256, int(settings.vlm_max_image_side))
    width, height = source.size
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        resized = source.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    else:
        resized = source

    buffer = BytesIO()
    resized.save(buffer, format="JPEG", quality=max(50, min(95, int(settings.vlm_image_jpeg_quality))), optimize=True)
    return image_to_base64(Image.open(BytesIO(buffer.getvalue())), mime_type="image/jpeg")


def parse_answer(content: str) -> FrameChatAnswer:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"conclusion": content.strip() or "답변을 생성하지 못했습니다."}
    if not isinstance(data, dict):
        data = {"conclusion": str(data)}

    reasoning = data.get("reasoning_summary")
    recommendations = data.get("recommendations")
    evidence_images = data.get("evidence_images")
    return FrameChatAnswer(
        conclusion=str(data.get("conclusion", "제공된 이미지 근거만으로는 결론이 제한적입니다.")),
        reasoning_summary=[str(item) for item in reasoning] if isinstance(reasoning, list) and reasoning else ["원본, Heatmap, Scanpath 이미지를 함께 확인했습니다."],
        evidence_images=[str(item) for item in evidence_images] if isinstance(evidence_images, list) else ["original", "heatmap_overlay", "scanpath_overlay"],
        risk_level=data.get("risk_level") if data.get("risk_level") in {"low", "medium", "high"} else "medium",
        recommendations=[str(item) for item in recommendations] if isinstance(recommendations, list) and recommendations else ["중요 CTA 주변의 시각적 경쟁 요소를 줄이는 방안을 검토하세요."],
        confidence=data.get("confidence") if data.get("confidence") in {"low", "medium", "high"} else "medium",
        caveat=str(data.get("caveat") or CHAT_CAVEAT),
    )


def stream_event(event: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"event": event, "data": data}


def safe_error_text(value: str) -> str:
    cleaned = " ".join(value.replace("\n", " ").split())
    if not cleaned:
        return "empty response body"
    return cleaned[:500]


def answer_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "conclusion": {"type": "string"},
            "reasoning_summary": {"type": "array", "items": {"type": "string"}},
            "evidence_images": {"type": "array", "items": {"type": "string"}},
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "caveat": {"type": "string"},
        },
        "required": ["conclusion", "reasoning_summary", "evidence_images", "risk_level", "recommendations", "confidence", "caveat"],
    }
