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
        payload = build_ollama_payload(settings=settings, messages=build_ollama_messages(request, settings, streaming=False))
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
            "progress",
            {
                "stage": "image_evidence",
                "message": "VLM에 원본/Heatmap/Scanpath 근거 이미지를 전달했습니다.",
                "evidence_images": image_roles(request),
                "progress": 0.25,
            },
        )
        payload = build_ollama_stream_payload(settings=settings, messages=build_ollama_messages(request, settings, streaming=True))
        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        async for item in stream_ollama_chat(
            client,
            settings,
            payload,
            image_count=len(request.selected_images[:MAX_FRAME_CHAT_IMAGES]),
            answer_parts=answer_parts,
            thinking_parts=thinking_parts,
        ):
            yield item
        answer = answer_from_stream_text("".join(answer_parts), "".join(thinking_parts), request)

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
    content: list[dict[str, Any]] = [{"type": "input_text", "text": build_prompt(request, streaming=False)}]
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


def build_ollama_stream_payload(*, settings: Settings, messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload = build_ollama_payload(settings=settings, messages=messages)
    payload["stream"] = True
    payload.pop("format", None)
    payload["think"] = False
    return payload


def build_ollama_messages(request: FrameChatRequest, settings: Settings, *, streaming: bool) -> list[dict[str, Any]]:
    message: dict[str, Any] = {"role": "user", "content": build_prompt(request, streaming=streaming)}
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


async def stream_ollama_chat(
    client: httpx.AsyncClient,
    settings: Settings,
    payload: dict[str, Any],
    *,
    image_count: int,
    answer_parts: list[str],
    thinking_parts: list[str],
):
    async with client.stream("POST", f"{settings.ollama_base_url.rstrip('/')}/api/chat", json=payload) as response:
        if response.status_code >= 400:
            text = await response.aread()
            raise RuntimeError(
                f"Ollama rejected frame chat stream status={response.status_code} "
                f"image_count={image_count} message={safe_error_text(text.decode('utf-8', errors='replace'))}"
            )
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = chunk.get("message") if isinstance(chunk.get("message"), dict) else {}
            thinking_delta = str(message.get("thinking") or chunk.get("thinking") or "")
            answer_delta = str(message.get("content") or "")
            if thinking_delta:
                thinking_parts.append(thinking_delta)
                yield stream_event("thinking_delta", {"delta": thinking_delta, "progress": 0.45})
            if answer_delta:
                answer_parts.append(answer_delta)
                yield stream_event("answer_delta", {"delta": answer_delta, "progress": 0.7})
            if chunk.get("done"):
                break


def build_prompt(request: FrameChatRequest, *, streaming: bool) -> str:
    previous = [{"role": item.role, "content": item.content[:500]} for item in request.previous_messages[-6:]]
    common = (
        f"Question: {request.question}\n"
        f"Frame ID: {request.frame_id}\n"
        f"Frame name: {request.frame_name or request.frame_id}\n"
        f"Metrics JSON: {json.dumps(request.metrics, ensure_ascii=False)}\n"
        f"Image roles JSON: {json.dumps(image_roles(request), ensure_ascii=False)}\n"
        f"Previous messages JSON: {json.dumps(previous, ensure_ascii=False)}\n\n"
    )
    if streaming:
        return common + (
            "반드시 한국어로만 답하세요. 실제 사용자 테스트(UT)에 참여한 대상자처럼, 먼저 짧은 사용자 페르소나를 가정한 뒤 "
            "1인칭 관찰과 행동 가능성을 섞어 답하세요. 예: '저는 ... 사용자라고 가정하면 ...'. "
            "첨부된 원본 화면, Heatmap overlay, Scanpath overlay 이미지를 가장 중요한 근거로 사용하세요. "
            "Heatmap은 시선이 어디에 끌릴 가능성이 높은지, Scanpath는 시선 이동과 탐색 흐름이 어떤지 설명하세요. "
            "IA Flow, target path, memory blur, 휴리스틱 플로우 평가는 언급하지 마세요. "
            "자연스러운 한국어 문단으로 답하고, 마지막에는 사용자가 바로 수정할 수 있는 제안을 2-4개 포함하세요."
        )
    return common + (
        "Use the attached original screen, heatmap overlay, and scanpath overlay as primary evidence. "
        "Answer in Korean only. Start from a concise UT participant persona and explain what the heatmap suggests users notice, "
        "where the scanpath likely moves, and how that affects the user's question. "
        "Do not discuss IA Flow, target paths, memory blur, or heuristic flow evaluation. "
        "Return only JSON with conclusion, reasoning_summary, evidence_images, risk_level, recommendations, confidence, caveat."
    )


def system_prompt() -> str:
    return (
        "너는 하나의 Figma 프레임을 보는 UX Bot이다. "
        "항상 한국어로 답하고, 실제 UT 참가자가 화면을 보는 것처럼 짧은 페르소나를 가정해 말한다. "
        "원본 UI, 예측 Heatmap, Scanpath 근거에서 벗어난 추측은 줄이고 실무적으로 답한다."
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


def answer_from_stream_text(answer_text: str, thinking_text: str, request: FrameChatRequest) -> FrameChatAnswer:
    conclusion = answer_text.strip() or "제공된 이미지 근거만으로는 답변을 생성하지 못했습니다."
    reasoning = ["원본 화면, Heatmap overlay, Scanpath overlay 이미지를 함께 확인했습니다."]
    if thinking_text:
        reasoning.append("모델의 중간 추론 신호를 스트리밍으로 수신했습니다.")
    return FrameChatAnswer(
        conclusion=conclusion,
        reasoning_summary=reasoning,
        evidence_images=[item["image_role"] for item in image_roles(request)],
        risk_level="medium",
        recommendations=["Heatmap 집중 구역과 Scanpath가 지나치게 길어지는 구역을 우선 조정하세요."],
        confidence="medium",
        caveat=CHAT_CAVEAT,
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
