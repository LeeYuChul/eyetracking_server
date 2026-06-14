from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.flow import ChatMessage, FrameMetrics, VisualArtifact


class FrameInput(BaseModel):
    client_frame_id: str
    frame_name: str
    figma_node_id: str | None = None
    width: int | float | None = None
    height: int | float | None = None
    export_scale: int | float | None = None
    file_key: str | None = None
    order_index: int = 0


class FrameAnalysisResult(BaseModel):
    client_frame_id: str
    figma_node_id: str | None = None
    frame_name: str
    width: int
    height: int
    order_index: int
    metrics: FrameMetrics
    artifacts: dict[str, VisualArtifact]


class FrameModelInfo(BaseModel):
    heatmap_model: str
    heatmap_version: str
    heatmap_backend: str
    scanpath_model: str = "deterministic-saliency-peaks"


class FrameAnalysisBundle(BaseModel):
    analysis_bundle_id: str
    created_at: str
    storage_policy: str = "client_must_store_response"
    frames: list[FrameAnalysisResult]
    model_info: FrameModelInfo


class FrameImageInput(BaseModel):
    artifact_type: str
    mime_type: str = "image/png"
    base64: str
    width: int | None = None
    height: int | None = None
    encoding: str = "base64"
    frame_id: str | None = None
    frame_name: str | None = None
    image_role: Literal["original", "heatmap_overlay", "scanpath_overlay"] | str | None = None


class FrameChatRequest(BaseModel):
    question: str
    frame_id: str
    frame_name: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    selected_images: list[FrameImageInput]
    previous_messages: list[ChatMessage] = Field(default_factory=list)


class FrameChatAnswer(BaseModel):
    conclusion: str
    reasoning_summary: list[str]
    evidence_images: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    recommendations: list[str]
    confidence: Literal["low", "medium", "high"] = "medium"
    caveat: str


class FrameChatResponse(BaseModel):
    answer: FrameChatAnswer
    storage_policy: str = "client_must_store_chat_if_needed"
    provider: str
    model: str
