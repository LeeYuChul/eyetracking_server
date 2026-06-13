from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.config import Settings
from app.models.flow import UxAnswer, UxEvaluateRequest, UxEvaluateResponse
from app.services.artifacts import artifact_to_data_url

UX_CAVEAT = "이 답변은 예측 시선/메모리 모델 기반 참고 결과이며 실제 사용자 테스트를 대체하지 않습니다."
MAX_VLM_IMAGES = 20


async def evaluate_ux(request: UxEvaluateRequest, settings: Settings) -> UxEvaluateResponse:
    provider = settings.vlm_provider.strip().lower()
    if provider == "openai":
        answer = await evaluate_with_openai(request, settings)
        return UxEvaluateResponse(answer=answer, provider="openai", model=settings.openai_model)
    if provider == "ollama":
        answer = await evaluate_with_ollama(request, settings)
        return UxEvaluateResponse(answer=answer, provider="ollama", model=settings.ollama_model)
    raise RuntimeError(f"Unsupported VLM provider: {settings.vlm_provider}")


def evaluate_with_heuristic(request: UxEvaluateRequest) -> UxEvaluateResponse:
    answer = build_heuristic_answer(request)
    return UxEvaluateResponse(answer=answer, provider="heuristic", model="ux-flow-heuristic-chat-0.1")


async def evaluate_with_ollama(request: UxEvaluateRequest, settings: Settings) -> UxAnswer:
    images = selected_images(request)
    if not images:
        raise RuntimeError("VLM evaluation requires selected_images")
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "stream": False,
        "messages": build_ollama_messages(request),
        "format": "json",
        "keep_alive": settings.ollama_keep_alive,
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(f"{settings.ollama_base_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
    data = response.json()
    content = data.get("message", {}).get("content", "")
    return parse_answer(content, request)


def build_heuristic_answer(request: UxEvaluateRequest) -> UxAnswer:
    evidence = compact_evidence(request.evidence)
    target_result = evidence.get("target_result", {})
    frames = evidence.get("frames", [])
    target_frames = target_result.get("frames", []) if isinstance(target_result, dict) else []
    path_frame_ids = target_result.get("path_frame_ids", []) if isinstance(target_result, dict) else []

    avg_retention = None
    max_distance = 0
    if isinstance(target_frames, list) and target_frames:
        retentions = []
        for frame in target_frames:
            if not isinstance(frame, dict):
                continue
            metrics = frame.get("memory_metrics")
            if isinstance(metrics, dict):
                value = metrics.get("estimated_retention")
                if isinstance(value, (int, float)):
                    retentions.append(float(value))
                distance = metrics.get("temporal_distance")
                if isinstance(distance, int):
                    max_distance = max(max_distance, distance)
        if retentions:
            avg_retention = sum(retentions) / len(retentions)

    cognitive_load = "medium"
    if isinstance(frames, list) and frames:
        path_lengths = []
        complexities = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            metrics = frame.get("metrics")
            if isinstance(metrics, dict):
                if isinstance(metrics.get("scanpath_length"), (int, float)):
                    path_lengths.append(float(metrics["scanpath_length"]))
                if isinstance(metrics.get("visual_complexity"), (int, float)):
                    complexities.append(float(metrics["visual_complexity"]))
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0.0
        avg_path = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0
        if avg_complexity > 0.62 or avg_path > 1200:
            cognitive_load = "high"
        elif avg_complexity < 0.35 and avg_path < 700:
            cognitive_load = "low"

    if avg_retention is None:
        conclusion = "제공된 근거만으로는 기억 가능성을 단정하기 어렵지만, Flow 구조와 시선 지표 기준으로 추가 확인이 필요합니다."
        risk_level = "medium"
        confidence = "low"
    elif avg_retention < 0.35 or cognitive_load == "high":
        conclusion = "현재 Flow에서는 사용자가 이전 화면의 핵심 정보를 기억하거나 목표 행동 단서를 찾을 가능성이 낮아 보입니다."
        risk_level = "high"
        confidence = "medium"
    elif avg_retention > 0.62 and cognitive_load != "high":
        conclusion = "현재 근거 기준으로는 목표 화면까지 필요한 정보를 유지하고 행동할 가능성이 비교적 높습니다."
        risk_level = "low"
        confidence = "medium"
    else:
        conclusion = "기억 가능성과 목표 행동 가능성은 중간 수준으로 보이며, 목표 화면에서 보조 단서를 제공하는 편이 안전합니다."
        risk_level = "medium"
        confidence = "medium"

    reasoning = [
        f"Target path는 {len(path_frame_ids) if isinstance(path_frame_ids, list) else 0}개 프레임으로 구성되어 있습니다.",
        f"인지 부하 추정치는 {cognitive_load} 수준입니다.",
    ]
    if avg_retention is not None:
        reasoning.append(f"이전 화면의 평균 기억 유지 추정치는 {avg_retention:.2f}입니다.")
    if max_distance:
        reasoning.append(f"가장 먼 이전 화면의 temporal distance는 {max_distance}입니다.")

    recommendations = [
        "목표 화면에서 이전 화면의 핵심 정보나 선택 내용을 다시 노출하세요.",
        "CTA 주변의 시각적 경쟁 요소를 줄이고, Scanpath가 길어지는 구간을 우선 검토하세요.",
    ]
    if cognitive_load == "high":
        recommendations.append("복잡도가 높은 중간 화면에는 단계 표시나 요약 정보를 추가하는 것이 좋습니다.")

    evidence_frames = []
    if isinstance(path_frame_ids, list):
        evidence_frames = [str(frame_id) for frame_id in path_frame_ids[-4:]]
    if not evidence_frames:
        evidence_frames = [request.target_frame_id]

    return UxAnswer(
        conclusion=conclusion,
        reasoning_summary=reasoning,
        evidence_frames=evidence_frames,
        risk_level=risk_level,
        recommendations=recommendations,
        confidence=confidence,
        caveat=UX_CAVEAT,
    )


async def evaluate_with_openai(request: UxEvaluateRequest, settings: Settings) -> UxAnswer:
    if not settings.openai_api_key:
        raise RuntimeError("OpenAI API key is not configured")
    if not selected_images(request):
        raise RuntimeError("VLM evaluation requires selected_images")

    content: list[dict[str, Any]] = [{"type": "input_text", "text": build_prompt(request)}]
    for image in selected_images(request)[:MAX_VLM_IMAGES]:
        content.append({"type": "input_image", "image_url": artifact_to_data_url(image)})

    payload = {
        "model": settings.openai_model,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ux_evaluation",
                "schema": answer_schema(),
                "strict": True,
            }
        },
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(f"{settings.openai_base_url.rstrip('/')}/responses", json=payload, headers=headers)
        response.raise_for_status()
    data = response.json()
    return parse_answer(data.get("output_text", ""), request)


def build_ollama_messages(request: UxEvaluateRequest) -> list[dict[str, Any]]:
    message: dict[str, Any] = {"role": "user", "content": build_prompt(request)}
    images = selected_images(request)[:MAX_VLM_IMAGES]
    if images:
        message["images"] = [image.base64.split(",", 1)[-1] for image in images]
    return [{"role": "system", "content": system_prompt()}, message]


def build_prompt(request: UxEvaluateRequest) -> str:
    evidence = compact_evidence(request.evidence)
    previous = [{"role": item.role, "content": item.content[:500]} for item in request.previous_messages[-4:]]
    image_context = selected_image_context(request.evidence)
    return (
        f"{system_prompt()}\n\n"
        f"Question: {request.question}\n"
        f"Target frame: {request.target_frame_id}\n"
        f"Previous messages summary JSON: {json.dumps(previous, ensure_ascii=False)}\n"
        f"Evidence JSON: {json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"Attached image context JSON: {json.dumps(image_context, ensure_ascii=False)}\n\n"
        "Answer from the target frame user's perspective, after reviewing the entire IA Flow and every attached screenshot. "
        "If the question asks what button to press, identify the visible button, label, tab, menu item, or CTA in the target frame "
        "and describe its approximate location. Use previous frames, memory blur, heatmap, scanpath, and metrics as supporting evidence only. "
        "Do not answer only with memory-retention metrics or a generic template. "
        "Return only JSON with conclusion, reasoning_summary, evidence_frames, risk_level, recommendations, confidence, caveat."
    )


def system_prompt() -> str:
    return (
        "You are an image-based UX flow reviewer for Figma mobile screens. The attached images are primary evidence. "
        "Visually inspect text labels, buttons, icons, navigation areas, and CTA placement. Use heatmap, scanpath, "
        "memory blur, IA Flow order, and flow metrics only as supporting evidence. Prioritize the requested target frame. Do not claim real user testing. "
        "Keep reasoning user-visible and concise."
    )


def compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    frames = evidence.get("frames")
    if isinstance(frames, list):
        compact["frames"] = [
            {
                "client_frame_id": frame.get("client_frame_id"),
                "frame_name": frame.get("frame_name"),
                "metrics": frame.get("metrics") or frame.get("scanpath_metrics"),
                "parsed": frame.get("parsed"),
            }
            for frame in frames
            if isinstance(frame, dict)
        ][:15]
    target_result = evidence.get("target_result")
    if isinstance(target_result, dict):
        compact["target_result"] = {
            "target_frame_id": target_result.get("target_frame_id"),
            "path_frame_ids": target_result.get("path_frame_ids"),
            "frames": [
                {
                    "client_frame_id": frame.get("client_frame_id"),
                    "temporal_distance": frame.get("temporal_distance"),
                    "memory_metrics": frame.get("memory_metrics"),
                }
                for frame in target_result.get("frames", [])
                if isinstance(frame, dict)
            ],
        }
    compact["target_frame"] = frame_summary_by_id(compact.get("frames", []), target_result.get("target_frame_id") if isinstance(target_result, dict) else None)
    compact["selected_image_count"] = len(selected_images_from_raw(evidence))
    compact["selected_images"] = selected_image_context(evidence)
    return compact


def frame_summary_by_id(frames: Any, frame_id: Any) -> dict[str, Any] | None:
    if not isinstance(frames, list) or not isinstance(frame_id, str):
        return None
    for frame in frames:
        if isinstance(frame, dict) and frame.get("client_frame_id") == frame_id:
            return frame
    return None


def selected_image_context(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    context = []
    raw_images = evidence.get("selected_images", [])
    if not isinstance(raw_images, list):
        return context
    for index, item in enumerate(raw_images):
        if not isinstance(item, dict):
            continue
        context.append(
            {
                "index": index,
                "artifact_type": item.get("artifact_type"),
                "frame_id": item.get("frame_id"),
                "frame_name": item.get("frame_name"),
                "width": item.get("width"),
                "height": item.get("height"),
            }
        )
    return context


def selected_images(request: UxEvaluateRequest):
    return selected_images_from_raw(request.evidence)


def selected_images_from_raw(evidence: dict[str, Any]):
    from app.models.flow import ImageArtifactInput

    images = []
    raw_images = evidence.get("selected_images", [])
    if not isinstance(raw_images, list):
        return images
    for item in raw_images:
        if not isinstance(item, dict) or "base64" not in item:
            continue
        images.append(ImageArtifactInput(**item))
    return images


def parse_answer(content: str, request: UxEvaluateRequest) -> UxAnswer:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"conclusion": content.strip() or "평가 결과를 생성하지 못했습니다."}
    if not isinstance(data, dict):
        data = {"conclusion": str(data)}

    reasoning = data.get("reasoning_summary")
    recommendations = data.get("recommendations")
    return UxAnswer(
        conclusion=str(data.get("conclusion", "제공된 근거만으로는 결론이 제한적입니다.")),
        reasoning_summary=[str(item) for item in reasoning] if isinstance(reasoning, list) and reasoning else ["제공된 flow, heatmap, scanpath, memory 지표를 종합했습니다."],
        evidence_frames=[str(item) for item in data.get("evidence_frames", [])] if isinstance(data.get("evidence_frames"), list) else [request.target_frame_id],
        risk_level=data.get("risk_level") if data.get("risk_level") in {"low", "medium", "high"} else "medium",
        recommendations=[str(item) for item in recommendations] if isinstance(recommendations, list) and recommendations else ["목표 프레임에서 핵심 단서와 CTA를 다시 노출하는 방안을 검토하세요."],
        confidence=data.get("confidence") if data.get("confidence") in {"low", "medium", "high"} else "medium",
        caveat=str(data.get("caveat") or UX_CAVEAT),
    )


def answer_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "conclusion": {"type": "string"},
            "reasoning_summary": {"type": "array", "items": {"type": "string"}},
            "evidence_frames": {"type": "array", "items": {"type": "string"}},
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "caveat": {"type": "string"},
        },
        "required": ["conclusion", "reasoning_summary", "evidence_frames", "risk_level", "recommendations", "confidence", "caveat"],
    }
