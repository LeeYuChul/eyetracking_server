from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WarningItem(BaseModel):
    code: str
    message: str
    client_frame_ids: list[str] = Field(default_factory=list)


class ClientFrameInput(BaseModel):
    client_frame_id: str
    frame_name: str
    figma_node_id: str | None = None
    width: int | float | None = None
    height: int | float | None = None
    export_scale: int | float | None = None
    file_key: str | None = None
    order_index: int = 0


class ParsedFrame(BaseModel):
    client_frame_id: str
    frame_name: str
    flow_id: str | None = None
    depth: int | None = None
    state: int | None = None
    screen_name: str | None = None
    parse_status: Literal["parsed", "unparsed"] = "unparsed"
    order_index: int = 0


class FlowFrameNode(BaseModel):
    client_frame_id: str
    frame_name: str
    depth: int | None = None
    state: int | None = None
    screen_name: str | None = None
    children: list["FlowFrameNode"] = Field(default_factory=list)


class FlowGroup(BaseModel):
    flow_id: str
    frames: list[FlowFrameNode]
    ordered_frame_ids: list[str]


class FlowTree(BaseModel):
    flows: list[FlowGroup]
    unparsed_frame_ids: list[str] = Field(default_factory=list)
    ordered_frame_ids: list[str] = Field(default_factory=list)


class FlowParseRequest(BaseModel):
    frames: list[ClientFrameInput]


class FlowParseResponse(BaseModel):
    flow_tree: FlowTree
    parsed_frames: list[ParsedFrame]
    warnings: list[WarningItem] = Field(default_factory=list)


class VisualArtifact(BaseModel):
    artifact_type: str
    mime_type: str = "image/png"
    base64: str
    width: int
    height: int
    encoding: Literal["base64", "data_url"] = "base64"


class FixationPoint(BaseModel):
    index: int
    x: int
    y: int
    score: float


class FrameMetrics(BaseModel):
    scanpath_length: float
    fixation_count: int
    attention_entropy: float
    visual_complexity: float
    fixations: list[FixationPoint]


class FrameAnalysisResult(BaseModel):
    client_frame_id: str
    figma_node_id: str | None = None
    frame_name: str
    parsed: ParsedFrame
    width: int
    height: int
    order_index: int
    metrics: FrameMetrics
    artifacts: dict[str, VisualArtifact]


class ModelInfo(BaseModel):
    heatmap_model: str
    heatmap_version: str
    heatmap_backend: str
    scanpath_model: str = "deterministic-saliency-peaks"


class AnalysisBundle(BaseModel):
    analysis_bundle_id: str
    created_at: str
    storage_policy: str = "client_must_store_response"
    flow_tree: FlowTree
    frames: list[FrameAnalysisResult]
    warnings: list[WarningItem] = Field(default_factory=list)
    model_info: ModelInfo


class ImageArtifactInput(BaseModel):
    mime_type: str = "image/png"
    base64: str
    width: int | None = None
    height: int | None = None
    encoding: str = "base64"


class TargetFrameInput(BaseModel):
    client_frame_id: str
    frame_name: str | None = None
    original_image: ImageArtifactInput | None = None
    heatmap: ImageArtifactInput | None = None
    heatmap_overlay: ImageArtifactInput | None = None
    scanpath_overlay: ImageArtifactInput | None = None
    scanpath_metrics: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    parsed: dict[str, Any] | None = None


class PrepareTargetRequest(BaseModel):
    target_frame_id: str
    flow_tree: FlowTree
    frames: list[TargetFrameInput]
    options: dict[str, Any] = Field(default_factory=dict)


class MemoryMetrics(BaseModel):
    estimated_retention: float
    blur_strength_avg: float
    temporal_distance: int
    depth_blur_strength: float | None = None
    cumulative_blur_strength: float | None = None


class TargetFrameResult(BaseModel):
    client_frame_id: str
    temporal_distance: int
    memory_metrics: MemoryMetrics
    artifacts: dict[str, VisualArtifact]


class TargetResult(BaseModel):
    target_result_id: str
    target_frame_id: str
    path_frame_ids: list[str]
    frames: list[TargetFrameResult]
    memory_model_options: dict[str, Any]
    created_at: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class UxEvaluateRequest(BaseModel):
    question: str
    target_frame_id: str
    flow_tree: FlowTree | dict[str, Any]
    evidence: dict[str, Any] = Field(default_factory=dict)
    previous_messages: list[ChatMessage] = Field(default_factory=list)


class UxAnswer(BaseModel):
    conclusion: str
    reasoning_summary: list[str]
    evidence_frames: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    recommendations: list[str]
    confidence: Literal["low", "medium", "high"] = "medium"
    caveat: str


class UxEvaluateResponse(BaseModel):
    answer: UxAnswer
    storage_policy: str = "client_must_store_chat_if_needed"
    provider: str
    model: str


FlowFrameNode.model_rebuild()
