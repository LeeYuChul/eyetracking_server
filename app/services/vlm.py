from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from app.core.config import Settings
from app.models.frame import FrameChatAnswer, FrameChatRequest, FrameChatResponse
from app.services.artifacts import artifact_to_data_url, image_from_artifact, image_to_base64

CHAT_CAVEAT = "This answer is based on predictive eye-tracking heatmap/scanpath evidence and does not replace real usability testing."
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
            "message": "Reviewing the selected frame, heatmap, and scanpath evidence.",
            "progress": 0,
        },
    )
    async with httpx.AsyncClient(timeout=settings.vlm_request_timeout_seconds) as client:
        yield stream_event(
            "progress",
            {
                "stage": "image_evidence",
                "message": "Sending the original frame, heatmap overlay, and scanpath overlay to the VLM.",
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

    yield stream_event("progress", {"stage": "synthesizing", "message": "Synthesizing the visual evidence and attention metrics.", "progress": 0.85})
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
            "Answer in the same language as the user's Question. If the Question mixes languages, use the dominant language of the Question. "
            "The most important rule is to answer the user's Question directly. "
            "Internal UX-review guidance, persona framing, heatmap evidence, and scanpath evidence are secondary and should only support the answer. "
            "If the Question asks about the meaning of a visible term, button, location, or state, answer that meaning clearly in the first 1-2 sentences. "
            "For example, if the user asks what a label means, explain the label's likely role/status on this screen before giving any UX critique. "
            "Include recommendations only when the Question explicitly asks for evaluation, improvement, heuristic analysis, or usability issues. "
            "You may speak from a lightweight usability-test participant perspective, but keep that persona brief and never let it override the Question. "
            "Use the attached original frame, heatmap overlay, and scanpath overlay as evidence, and skip unrelated eye-movement details. "
            "Do not discuss IA Flow, target paths, memory blur, or heuristic flow evaluation. "
            "Do not expand into topics the user did not ask about."
        )
    return common + (
        "Use the attached original screen, heatmap overlay, and scanpath overlay as primary evidence. "
        "Answer in the same language as the user's Question. The user's Question has priority over all internal UX-review instructions. "
        "Answer the Question directly first; use persona, heatmap, and scanpath only as supporting context. "
        "Include recommendations only if the Question asks for evaluation, improvement, heuristic analysis, or usability issues. "
        "Do not discuss IA Flow, target paths, memory blur, or heuristic flow evaluation. "
        "Return only JSON with conclusion, reasoning_summary, evidence_images, risk_level, recommendations, confidence, caveat."
    )


def system_prompt() -> str:
    return (
        "You are UX Bot reviewing one Figma frame at a time. "
        "Always prioritize the user's question and instructions over internal analysis templates. "
        "Answer in the same language as the user's question. "
        "The usability-test participant perspective is only a light supporting style; do not drift into unsolicited UX critique. "
        "Stay grounded in the original UI, predicted heatmap, and scanpath evidence."
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
        data = {"conclusion": content.strip() or "Could not generate an answer."}
    if not isinstance(data, dict):
        data = {"conclusion": str(data)}

    reasoning = data.get("reasoning_summary")
    recommendations = data.get("recommendations")
    evidence_images = data.get("evidence_images")
    return FrameChatAnswer(
        conclusion=str(data.get("conclusion", "The conclusion is limited by the provided image evidence.")),
        reasoning_summary=[str(item) for item in reasoning] if isinstance(reasoning, list) and reasoning else ["Reviewed the original frame, heatmap overlay, and scanpath overlay together."],
        evidence_images=[str(item) for item in evidence_images] if isinstance(evidence_images, list) else ["original", "heatmap_overlay", "scanpath_overlay"],
        risk_level=data.get("risk_level") if data.get("risk_level") in {"low", "medium", "high"} else "medium",
        recommendations=[str(item) for item in recommendations] if isinstance(recommendations, list) and recommendations else ["Review visual competition around important calls to action."],
        confidence=data.get("confidence") if data.get("confidence") in {"low", "medium", "high"} else "medium",
        caveat=str(data.get("caveat") or CHAT_CAVEAT),
    )


def answer_from_stream_text(answer_text: str, thinking_text: str, request: FrameChatRequest) -> FrameChatAnswer:
    conclusion = answer_text.strip() or "I could not generate an answer from the provided image evidence."
    reasoning = ["Reviewed the original frame, heatmap overlay, and scanpath overlay together."]
    if thinking_text:
        reasoning.append("Received intermediate model reasoning signals through the stream.")
    return FrameChatAnswer(
        conclusion=conclusion,
        reasoning_summary=reasoning,
        evidence_images=[item["image_role"] for item in image_roles(request)],
        risk_level="medium",
        recommendations=["Review areas with strong heatmap concentration and unusually long scanpath transitions first."],
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
