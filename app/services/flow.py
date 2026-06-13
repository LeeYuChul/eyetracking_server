from __future__ import annotations

import re
from collections import defaultdict

from app.models.flow import (
    ClientFrameInput,
    FlowFrameNode,
    FlowGroup,
    FlowParseResponse,
    FlowTree,
    ParsedFrame,
    WarningItem,
)

FRAME_NAME_RE = re.compile(r"^(?P<flow_id>[^_]+)_(?P<depth>\d+)(?:-(?P<state>\d+))?_(?P<screen_name>.+)$")


def parse_frame_name(frame: ClientFrameInput) -> ParsedFrame:
    match = FRAME_NAME_RE.match(frame.frame_name.strip())
    if not match:
        return ParsedFrame(
            client_frame_id=frame.client_frame_id,
            frame_name=frame.frame_name,
            parse_status="unparsed",
            order_index=frame.order_index,
        )

    return ParsedFrame(
        client_frame_id=frame.client_frame_id,
        frame_name=frame.frame_name,
        flow_id=match.group("flow_id"),
        depth=int(match.group("depth")),
        state=int(match.group("state")) if match.group("state") else None,
        screen_name=match.group("screen_name"),
        parse_status="parsed",
        order_index=frame.order_index,
    )


def build_flow_parse_response(frames: list[ClientFrameInput]) -> FlowParseResponse:
    parsed_frames = [parse_frame_name(frame) for frame in frames]
    warnings = build_warnings(parsed_frames)
    flow_tree = build_flow_tree(parsed_frames)
    return FlowParseResponse(flow_tree=flow_tree, parsed_frames=parsed_frames, warnings=warnings)


def build_warnings(parsed_frames: list[ParsedFrame]) -> list[WarningItem]:
    warnings: list[WarningItem] = []
    unparsed = [frame.client_frame_id for frame in parsed_frames if frame.parse_status != "parsed"]
    if unparsed:
        warnings.append(
            WarningItem(
                code="UNPARSED_FRAME_NAME",
                message="Some frame names do not match {flow}_{depth}_{screen} or {flow}_{depth}-{state}_{screen}.",
                client_frame_ids=unparsed,
            )
        )

    base_by_depth: dict[tuple[str, int], list[str]] = defaultdict(list)
    all_by_depth: dict[tuple[str, int], list[str]] = defaultdict(list)
    for frame in parsed_frames:
        if frame.parse_status != "parsed" or frame.flow_id is None or frame.depth is None:
            continue
        all_by_depth[(frame.flow_id, frame.depth)].append(frame.client_frame_id)
        if frame.state is None:
            base_by_depth[(frame.flow_id, frame.depth)].append(frame.client_frame_id)

    for ids in base_by_depth.values():
        if len(ids) > 1:
            warnings.append(
                WarningItem(
                    code="DUPLICATE_BASE_FRAME",
                    message="Multiple base frames share the same flow and depth.",
                    client_frame_ids=ids,
                )
            )

    for key, ids in all_by_depth.items():
        if len(ids) > 1 and not base_by_depth.get(key):
            warnings.append(
                WarningItem(
                    code="AMBIGUOUS_DEPTH",
                    message="A depth has state frames but no base frame.",
                    client_frame_ids=ids,
                )
            )

    return warnings


def build_flow_tree(parsed_frames: list[ParsedFrame]) -> FlowTree:
    flow_groups: list[FlowGroup] = []
    unparsed_ids = [frame.client_frame_id for frame in parsed_frames if frame.parse_status != "parsed"]

    by_flow: dict[str, list[ParsedFrame]] = defaultdict(list)
    for frame in parsed_frames:
        if frame.parse_status == "parsed" and frame.flow_id is not None:
            by_flow[frame.flow_id].append(frame)

    ordered_all: list[str] = []
    for flow_id in sorted(by_flow, key=natural_key):
        ordered = sorted(
            by_flow[flow_id],
            key=lambda item: (
                item.depth or 0,
                -1 if item.state is None else item.state,
                item.order_index,
            ),
        )
        ordered_ids = [frame.client_frame_id for frame in ordered]
        ordered_all.extend(ordered_ids)
        nodes_by_depth: dict[int, list[FlowFrameNode]] = defaultdict(list)
        root_nodes: list[FlowFrameNode] = []
        base_by_depth: dict[int, FlowFrameNode] = {}

        for frame in ordered:
            node = FlowFrameNode(
                client_frame_id=frame.client_frame_id,
                frame_name=frame.frame_name,
                depth=frame.depth,
                state=frame.state,
                screen_name=frame.screen_name,
            )
            depth = frame.depth or 0
            if frame.state is not None and depth in base_by_depth:
                base_by_depth[depth].children.append(node)
                continue
            if frame.state is None:
                base_by_depth[depth] = node
            parent_depths = [candidate for candidate in nodes_by_depth if candidate < depth]
            if parent_depths:
                parent_depth = max(parent_depths)
                nodes_by_depth[parent_depth][-1].children.append(node)
            else:
                root_nodes.append(node)
            nodes_by_depth[depth].append(node)

        flow_groups.append(FlowGroup(flow_id=flow_id, frames=root_nodes, ordered_frame_ids=ordered_ids))

    ordered_all.extend(unparsed_ids)
    return FlowTree(flows=flow_groups, unparsed_frame_ids=unparsed_ids, ordered_frame_ids=ordered_all)


def resolve_target_path(flow_tree: FlowTree, target_frame_id: str) -> list[str]:
    for flow in flow_tree.flows:
        if target_frame_id in flow.ordered_frame_ids:
            index = flow.ordered_frame_ids.index(target_frame_id)
            return flow.ordered_frame_ids[: index + 1]
    if target_frame_id in flow_tree.ordered_frame_ids:
        index = flow_tree.ordered_frame_ids.index(target_frame_id)
        return flow_tree.ordered_frame_ids[: index + 1]
    return []


def natural_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)
