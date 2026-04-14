"""
Workspace boundary scan core builders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np


@dataclass
class WorkspaceProfile:
    mode: int
    inner_segments: list[np.ndarray]
    outer_segments: list[np.ndarray]
    outer_open_curve_rz: np.ndarray
    unreachable_open_curves_rz: list[np.ndarray]
    closed_profile_rz: np.ndarray
    unreachable_closed_profiles_rz: list[np.ndarray]
    debug_data: dict | None = None


@dataclass
class BoundaryPrimitive:
    name: str
    points_rz: np.ndarray


@dataclass
class WorkspaceAnimationCurve:
    points_rz: np.ndarray
    label: str | None = None
    color: str | None = None
    linestyle: str = "-"
    linewidth: float = 2.0
    alpha: float = 1.0
    mirror: bool = True
    progressive: bool = True
    marker: str | None = None
    markersize: float = 6.0


@dataclass
class WorkspaceAnimationStage:
    title: str
    curves: list[WorkspaceAnimationCurve] = field(default_factory=list)
    annotation: str | None = None
    frames: int = 18
    state_sequence: list[dict[str, float]] = field(default_factory=list)


@dataclass
class WorkspaceAnimationData:
    mode: int
    stages: list[WorkspaceAnimationStage] = field(default_factory=list)
    csm_spec: dict[str, float] | None = None


@dataclass
class BoundaryScanOptions:
    length_samples: int = 240
    angle_samples: int = 240
    mode0_node_merge_tol: float = 2.0e-4
    mode0_endpoint_snap_tol: float = 3.0e-4
    mode0_route_length_samples: int = 72
    mode0_route_angle_samples: int = 72
    mode0_debug_output_dir: Path | None = None


def _tool_point_from_state(csm, *, mode, L1, L2, Lr, Ls, theta_1, theta_2):
    csm.set_state(
        mode=mode,
        phi=0.0,
        L1=float(L1),
        L2=float(L2),
        Lr=float(Lr),
        Ls=float(Ls),
        theta_1=float(theta_1),
        theta_2=float(theta_2),
        delta_1=0.0,
        delta_2=0.0,
    )
    return np.asarray(csm.pose[:3], dtype=float)


def _sample_state_curve(csm, states):
    points = np.asarray([_tool_point_from_state(csm, **state) for state in states], dtype=float)
    radii = np.hypot(points[:, 0], points[:, 1])
    return np.column_stack((radii, points[:, 2]))


def _build_axis_closure(start_rz, end_rz, samples=48):
    radii = np.linspace(float(start_rz[0]), float(end_rz[0]), samples)
    heights = np.linspace(float(start_rz[1]), float(end_rz[1]), samples)
    return np.column_stack((radii, heights))


def _close_curve_to_axis(curve_rz):
    curve_rz = np.asarray(curve_rz, dtype=float)
    if curve_rz.ndim != 2 or curve_rz.shape[0] < 2:
        return curve_rz
    start = curve_rz[0]
    end = curve_rz[-1]
    closure = np.asarray([[0.0, float(end[1])], [0.0, float(start[1])]], dtype=float)
    return np.vstack((curve_rz, closure))


def symmetric_fill_polygon(open_curve_rz):
    open_curve_rz = np.asarray(open_curve_rz, dtype=float)
    mirrored = np.column_stack((-open_curve_rz[:, 0], open_curve_rz[:, 1]))
    return np.vstack((mirrored[::-1], open_curve_rz[1:]))


def _cross_2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _segment_intersection(p0, p1, q0, q1, eps=1e-9):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    r = p1 - p0
    s = q1 - q0
    denom = _cross_2d(r, s)
    qmp = q0 - p0
    if abs(denom) < eps:
        return None

    t = _cross_2d(qmp, s) / denom
    u = _cross_2d(qmp, r) / denom
    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        point = p0 + np.clip(t, 0.0, 1.0) * r
        return float(t), float(u), point
    return None


def _project_point_to_segment(point, seg_start, seg_end, eps=1e-12):
    point = np.asarray(point, dtype=float)
    seg_start = np.asarray(seg_start, dtype=float)
    seg_end = np.asarray(seg_end, dtype=float)
    vec = seg_end - seg_start
    denom = float(np.dot(vec, vec))
    if denom <= eps:
        return 0.0, seg_start.copy(), float(np.linalg.norm(point - seg_start))
    t = float(np.clip(np.dot(point - seg_start, vec) / denom, 0.0, 1.0))
    proj = seg_start + t * vec
    dist = float(np.linalg.norm(point - proj))
    return t, proj, dist


def _boxes_overlap(min_a, max_a, min_b, max_b, pad=0.0):
    return not (
        max_a[0] < min_b[0] - pad or
        max_b[0] < min_a[0] - pad or
        max_a[1] < min_b[1] - pad or
        max_b[1] < min_a[1] - pad
    )


def _first_polyline_intersection(curve_a, curve_b):
    curve_a = np.asarray(curve_a, dtype=float)
    curve_b = np.asarray(curve_b, dtype=float)
    if curve_a.shape[0] < 2 or curve_b.shape[0] < 2:
        return None

    best = None
    best_progress = None
    for i in range(curve_a.shape[0] - 1):
        p0 = curve_a[i]
        p1 = curve_a[i + 1]
        for j in range(curve_b.shape[0] - 1):
            q0 = curve_b[j]
            q1 = curve_b[j + 1]
            result = _segment_intersection(p0, p1, q0, q1)
            if result is None:
                continue
            t, u, point = result
            progress = i + t
            if best is None or progress < best_progress:
                best = {
                    "idx_a": i,
                    "idx_b": j,
                    "t_a": t,
                    "t_b": u,
                    "point": np.asarray(point, dtype=float),
                }
                best_progress = progress
    return best


def _polyline_prefix(curve, split):
    curve = np.asarray(curve, dtype=float)
    if split is None:
        return curve.copy()
    idx = int(split["idx_a"])
    point = np.asarray(split["point"], dtype=float)
    prefix = curve[: idx + 1].copy()
    if prefix.shape[0] == 0 or not np.allclose(prefix[-1], point, atol=1e-9):
        prefix = np.vstack((prefix, point))
    else:
        prefix[-1] = point
    return prefix


def _polyline_suffix(curve, split, which="b"):
    curve = np.asarray(curve, dtype=float)
    if split is None:
        return curve.copy()
    idx_key = "idx_b" if which == "b" else "idx_a"
    idx = int(split[idx_key])
    point = np.asarray(split["point"], dtype=float)
    suffix = curve[idx + 1 :].copy()
    if suffix.shape[0] == 0:
        return point[np.newaxis, :]
    if not np.allclose(suffix[0], point, atol=1e-9):
        suffix = np.vstack((point, suffix))
    else:
        suffix[0] = point
    return suffix


def _concat_curve_segments(segments):
    cleaned = []
    for segment in segments:
        arr = np.asarray(segment, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        if not cleaned:
            cleaned.append(arr.copy())
            continue
        prev = cleaned[-1]
        if np.allclose(prev[-1], arr[0], atol=1e-9):
            cleaned.append(arr[1:].copy())
        else:
            cleaned.append(arr.copy())
    if not cleaned:
        return np.empty((0, 2), dtype=float)
    return np.vstack(cleaned)


def _extend_curve_outer_endpoint_downward(curve_rz, min_z):
    curve_rz = np.asarray(curve_rz, dtype=float)
    if curve_rz.ndim != 2 or curve_rz.shape[0] == 0:
        return curve_rz

    start_point = curve_rz[0].copy()
    end_point = curve_rz[-1].copy()
    extend_start = start_point[0] >= end_point[0]
    outer_point = start_point if extend_start else end_point
    target_z = float(min_z)
    if outer_point[1] <= target_z + 1e-9:
        return curve_rz

    extended_point = np.asarray([outer_point[0], target_z], dtype=float)
    if extend_start:
        if curve_rz.shape[0] >= 2 and np.allclose(curve_rz[1], extended_point, atol=1e-9):
            return curve_rz[1:].copy()
        return np.vstack((extended_point, curve_rz))

    if curve_rz.shape[0] >= 2 and np.allclose(curve_rz[-2], extended_point, atol=1e-9):
        return curve_rz[:-1].copy()
    return np.vstack((curve_rz, extended_point))


def _curve_progress_point(curve, progress):
    curve = np.asarray(curve, dtype=float)
    if curve.shape[0] == 0:
        return np.zeros(2, dtype=float)
    progress = float(np.clip(progress, 0.0, max(curve.shape[0] - 1, 0)))
    idx = int(np.floor(progress))
    frac = progress - idx
    if frac <= 1e-12 or idx >= curve.shape[0] - 1:
        return curve[min(idx, curve.shape[0] - 1)].copy()
    return curve[idx] + frac * (curve[idx + 1] - curve[idx])


def _extract_subcurve_by_progress(curve, progress_start, progress_end):
    curve = np.asarray(curve, dtype=float)
    start = float(progress_start)
    end = float(progress_end)
    if end < start:
        return _extract_subcurve_by_progress(curve, end, start)[::-1].copy()

    start_point = _curve_progress_point(curve, start)
    end_point = _curve_progress_point(curve, end)
    start_idx = int(np.floor(start))
    end_idx = int(np.floor(end))

    points = [start_point]
    for idx in range(start_idx + 1, end_idx + 1):
        if idx < curve.shape[0]:
            points.append(curve[idx].copy())
    if not np.allclose(points[-1], end_point, atol=1e-9):
        points.append(end_point)
    else:
        points[-1] = end_point
    return np.asarray(points, dtype=float)


def _dedupe_progress_points(items, tol=1e-9):
    if not items:
        return []

    items = sorted(items, key=lambda item: item[0])
    deduped = [items[0]]
    for progress, point in items[1:]:
        prev_progress, prev_point = deduped[-1]
        if abs(progress - prev_progress) <= tol:
            merged_point = 0.5 * (np.asarray(prev_point, dtype=float) + np.asarray(point, dtype=float))
            deduped[-1] = (0.5 * (prev_progress + progress), merged_point)
        else:
            deduped.append((progress, np.asarray(point, dtype=float)))
    return deduped


def _split_curve_at_intersections(curves, endpoint_snap_tol):
    split_points = []
    curve_boxes = []
    for curve in curves:
        curve_points = np.asarray(curve["points"], dtype=float)
        points = [(float(idx), np.asarray(point, dtype=float)) for idx, point in enumerate(curve_points)]
        split_points.append(points)
        if curve_points.shape[0] == 0:
            curve_boxes.append(None)
        else:
            curve_boxes.append((np.min(curve_points, axis=0), np.max(curve_points, axis=0)))

    for idx_a, curve_a in enumerate(curves):
        points_a = np.asarray(curve_a["points"], dtype=float)
        if points_a.shape[0] < 2:
            continue
        for idx_b in range(idx_a + 1, len(curves)):
            points_b = np.asarray(curves[idx_b]["points"], dtype=float)
            if points_b.shape[0] < 2:
                continue
            bbox_a = curve_boxes[idx_a]
            bbox_b = curve_boxes[idx_b]
            if bbox_a is not None and bbox_b is not None and not _boxes_overlap(bbox_a[0], bbox_a[1], bbox_b[0], bbox_b[1], pad=endpoint_snap_tol):
                continue
            for seg_a in range(points_a.shape[0] - 1):
                seg_a_min = np.minimum(points_a[seg_a], points_a[seg_a + 1])
                seg_a_max = np.maximum(points_a[seg_a], points_a[seg_a + 1])
                for seg_b in range(points_b.shape[0] - 1):
                    seg_b_min = np.minimum(points_b[seg_b], points_b[seg_b + 1])
                    seg_b_max = np.maximum(points_b[seg_b], points_b[seg_b + 1])
                    if not _boxes_overlap(seg_a_min, seg_a_max, seg_b_min, seg_b_max, pad=endpoint_snap_tol):
                        continue
                    hit = _segment_intersection(points_a[seg_a], points_a[seg_a + 1], points_b[seg_b], points_b[seg_b + 1])
                    if hit is None:
                        continue
                    t_a, t_b, point = hit
                    split_points[idx_a].append((seg_a + float(t_a), point))
                    split_points[idx_b].append((seg_b + float(t_b), point))

    for idx_a, curve_a in enumerate(curves):
        points_a = np.asarray(curve_a["points"], dtype=float)
        if points_a.shape[0] == 0:
            continue
        endpoint_progresses = [0.0, float(points_a.shape[0] - 1)]
        for endpoint_progress in endpoint_progresses:
            endpoint = _curve_progress_point(points_a, endpoint_progress)
            nearby_points = [endpoint]
            endpoint_box_min = endpoint - endpoint_snap_tol
            endpoint_box_max = endpoint + endpoint_snap_tol
            for idx_b, curve_b in enumerate(curves):
                if idx_a == idx_b:
                    continue
                points_b = np.asarray(curve_b["points"], dtype=float)
                if points_b.shape[0] < 2:
                    continue
                bbox_b = curve_boxes[idx_b]
                if bbox_b is not None and not _boxes_overlap(endpoint_box_min, endpoint_box_max, bbox_b[0], bbox_b[1], pad=0.0):
                    continue
                best = None
                for seg_b in range(points_b.shape[0] - 1):
                    seg_b_min = np.minimum(points_b[seg_b], points_b[seg_b + 1])
                    seg_b_max = np.maximum(points_b[seg_b], points_b[seg_b + 1])
                    if not _boxes_overlap(endpoint_box_min, endpoint_box_max, seg_b_min, seg_b_max, pad=0.0):
                        continue
                    t_b, proj, dist = _project_point_to_segment(endpoint, points_b[seg_b], points_b[seg_b + 1])
                    if dist > endpoint_snap_tol:
                        continue
                    candidate = (dist, seg_b + float(t_b), proj)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
                if best is not None:
                    _, progress_b, proj_b = best
                    nearby_points.append(proj_b)
                    split_points[idx_b].append((progress_b, proj_b))
            snapped_center = np.mean(np.asarray(nearby_points, dtype=float), axis=0)
            split_points[idx_a].append((endpoint_progress, snapped_center))

    return [_dedupe_progress_points(items) for items in split_points]


def _right_turn_angle_deg(vec_in, vec_out):
    angle_in = math.atan2(float(vec_in[1]), float(vec_in[0]))
    angle_out = math.atan2(float(vec_out[1]), float(vec_out[0]))
    return math.degrees((angle_in - angle_out) % (2.0 * math.pi))


def _trace_right_turn_path(curves, start_curve_name, target_curve_names, *, node_merge_tol, endpoint_snap_tol):
    if isinstance(target_curve_names, str):
        target_curve_names = {target_curve_names}
    else:
        target_curve_names = set(target_curve_names)
    split_points = _split_curve_at_intersections(curves, endpoint_snap_tol=endpoint_snap_tol)

    node_points = []
    node_members = []

    def get_node_id(point):
        point = np.asarray(point, dtype=float)
        for node_id, center in enumerate(node_points):
            if np.linalg.norm(point - center) <= node_merge_tol:
                node_members[node_id].append(point.copy())
                node_points[node_id] = np.mean(np.asarray(node_members[node_id], dtype=float), axis=0)
                return node_id
        node_id = len(node_points)
        node_points.append(point.copy())
        node_members.append([point.copy()])
        return node_id

    adjacency = {}
    start_curve_nodes = []
    target_nodes = set()

    for curve_idx, curve in enumerate(curves):
        progress_items = split_points[curve_idx]
        for item_idx in range(len(progress_items) - 1):
            progress_a, _ = progress_items[item_idx]
            progress_b, _ = progress_items[item_idx + 1]
            if progress_b - progress_a <= 1e-9:
                continue
            subcurve = _extract_subcurve_by_progress(curve["points"], progress_a, progress_b)
            if subcurve.shape[0] < 2:
                continue

            node_a = get_node_id(subcurve[0])
            node_b = get_node_id(subcurve[-1])
            if node_a == node_b:
                continue

            edge_meta = {
                "curve_name": curve["name"],
                "curve_role": curve["role"],
                "is_target": curve["name"] in target_curve_names,
            }
            adjacency.setdefault(node_a, []).append({"to": node_b, "points": subcurve, "meta": edge_meta})
            adjacency.setdefault(node_b, []).append({"to": node_a, "points": subcurve[::-1].copy(), "meta": edge_meta})

            if curve["name"] == start_curve_name:
                start_curve_nodes.extend([node_a, node_b])
            if curve["name"] in target_curve_names:
                target_nodes.update((node_a, node_b))

    if not start_curve_nodes:
        raise RuntimeError(f"Unable to locate start curve '{start_curve_name}' in mode0 graph.")
    if not target_nodes:
        raise RuntimeError(f"Unable to locate target curve(s) '{sorted(target_curve_names)}' in mode0 graph.")

    start_node = min(set(start_curve_nodes), key=lambda node_id: node_points[node_id][0])
    start_edges = [
        edge for edge in adjacency.get(start_node, [])
        if edge["meta"]["curve_name"] == start_curve_name and edge["points"][-1, 0] > edge["points"][0, 0] + 1e-9
    ]
    if not start_edges:
        start_edges = [edge for edge in adjacency.get(start_node, []) if edge["meta"]["curve_name"] == start_curve_name]
    if not start_edges:
        raise RuntimeError("Unable to choose an outward-going edge on the mode1 inner contour for mode0.")

    current_edge = max(start_edges, key=lambda edge: edge["points"][-1, 0] - edge["points"][0, 0])
    initial_segment = current_edge["points"]

    def ordered_outgoing(node_id, prev_id, incoming_vec, visited_edges):
        outgoing = []
        for edge in adjacency.get(node_id, []):
            edge_key = (node_id, edge["to"])
            if edge_key in visited_edges:
                continue
            if edge["to"] == prev_id and len(adjacency.get(node_id, [])) > 1:
                continue

            outgoing_vec = edge["points"][1] - edge["points"][0]
            turn = _right_turn_angle_deg(incoming_vec, outgoing_vec)
            if 1e-6 < turn < 180.0 - 1e-6:
                priority = (0, -turn)
            elif turn <= 1e-6 or turn >= 360.0 - 1e-6:
                priority = (1, 0.0)
            elif abs(turn - 180.0) <= 1e-6:
                priority = (3, 0.0)
            else:
                priority = (2, min(turn - 180.0, 360.0 - turn))
            outgoing.append((priority, edge))
        outgoing.sort(key=lambda item: item[0])
        return [edge for _, edge in outgoing]

    dead_ends = []

    def search(current_node, prev_node, incoming_vec, visited_edges, path_segments):
        if current_node in target_nodes:
            return path_segments

        ordered_edges = ordered_outgoing(current_node, prev_node, incoming_vec, visited_edges)
        if not ordered_edges:
            dead_ends.append(
                {
                    "node": int(current_node),
                    "prev_node": None if prev_node is None else int(prev_node),
                    "point": node_points[current_node].copy(),
                    "degree": len(adjacency.get(current_node, [])),
                }
            )
        for edge in ordered_edges:
            edge_key = (current_node, edge["to"])
            updated_path = path_segments + [edge["points"]]
            result = search(
                edge["to"],
                current_node,
                edge["points"][-1] - edge["points"][-2],
                visited_edges | {edge_key},
                updated_path,
            )
            if result is not None:
                return result
        return None

    path_segments = search(
        current_edge["to"],
        start_node,
        initial_segment[-1] - initial_segment[-2],
        {(start_node, current_edge["to"])},
        [initial_segment],
    )
    if path_segments is None:
        raise RuntimeError("Mode0 path tracing could not reach the outer contour after exploring all right-turn branches.")

    return _concat_curve_segments(path_segments), {
        "node_points": [point.copy() for point in node_points],
        "node_degrees": {int(node): len(edges) for node, edges in adjacency.items()},
        "num_nodes": int(len(node_points)),
        "num_directed_edges": int(sum(len(edges) for edges in adjacency.values())),
        "start_node": int(start_node),
        "target_nodes": sorted(target_nodes),
        "path_segments": [segment.copy() for segment in path_segments],
        "dead_ends": dead_ends,
    }


def _collect_mode0_sources(csm, options: BoundaryScanOptions):
    profiles = {
        1: build_mode1_profile(csm, options),
        2: build_mode2_profile(csm, options),
        3: build_mode3_profile(csm, options),
    }
    try:
        profiles[4] = build_mode4_profile(csm, options)
    except Exception:
        profiles[4] = None
    return profiles


def _write_mode0_debug_report(curves, error_message, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"mode0 trace failure: {error_message}", ""]
    lines.append("curve endpoints (meters):")
    endpoint_records = []
    for curve in curves:
        points = np.asarray(curve["points"], dtype=float)
        if points.shape[0] == 0:
            continue
        start = points[0]
        end = points[-1]
        endpoint_records.append((curve["name"], "start", start))
        endpoint_records.append((curve["name"], "end", end))
        lines.append(
            f"- {curve['name']} [{curve['role']}] start=({start[0]:.6f}, {start[1]:.6f}) "
            f"end=({end[0]:.6f}, {end[1]:.6f})"
        )

    lines.append("")
    lines.append("closest endpoint pairs:")
    pairs = []
    for idx, (name_a, tag_a, point_a) in enumerate(endpoint_records):
        for name_b, tag_b, point_b in endpoint_records[idx + 1 :]:
            if name_a == name_b:
                continue
            dist = float(np.linalg.norm(point_a - point_b))
            pairs.append((dist, name_a, tag_a, point_a, name_b, tag_b, point_b))
    for dist, name_a, tag_a, point_a, name_b, tag_b, point_b in sorted(pairs, key=lambda item: item[0])[:20]:
        lines.append(
            f"- {dist*1000.0:.3f} mm: {name_a}.{tag_a} ({point_a[0]:.6f}, {point_a[1]:.6f}) "
            f"<-> {name_b}.{tag_b} ({point_b[0]:.6f}, {point_b[1]:.6f})"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_mode1_inner_curve(csm, length_samples=240):
    inner_states = []
    for L2 in np.linspace(0.0, csm.L_20, length_samples):
        inner_states.append(
            {
                "mode": 1,
                "L1": 0.0,
                "L2": L2,
                "Lr": 0.0,
                "Ls": 0.0,
                "theta_1": 0.0,
                "theta_2": min(csm.kappa_20 * L2, csm.theta2_limit),
            }
        )
    return _sample_state_curve(csm, inner_states)


def _sample_mode2_theta_sweep_curve(csm, Lr, angle_samples=240):
    states = []
    for theta_2 in np.linspace(csm.theta2_limit, 0.0, angle_samples):
        states.append(
            {
                "mode": 2,
                "L1": 0.0,
                "L2": csm.L_20,
                "Lr": Lr,
                "Ls": 0.0,
                "theta_1": 0.0,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode2_theta_curve(csm, Lr, theta_start, theta_end, angle_samples=240):
    states = []
    for theta_2 in np.linspace(theta_start, theta_end, angle_samples):
        states.append(
            {
                "mode": 2,
                "L1": 0.0,
                "L2": csm.L_20,
                "Lr": Lr,
                "Ls": 0.0,
                "theta_1": 0.0,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode2_lr_curve(csm, theta_2, lr_start, lr_end, length_samples=240):
    start_point = _sample_state_curve(
        csm,
        [{"mode": 2, "L1": 0.0, "L2": csm.L_20, "Lr": lr_start, "Ls": 0.0, "theta_1": 0.0, "theta_2": theta_2}],
    )[0]
    end_point = _sample_state_curve(
        csm,
        [{"mode": 2, "L1": 0.0, "L2": csm.L_20, "Lr": lr_end, "Ls": 0.0, "theta_1": 0.0, "theta_2": theta_2}],
    )[0]
    radii = np.linspace(float(start_point[0]), float(end_point[0]), length_samples)
    heights = np.linspace(float(start_point[1]), float(end_point[1]), length_samples)
    return np.column_stack((radii, heights))


def _sample_mode3_theta2_curve(csm, L1, theta_1, theta_start, theta_end, angle_samples=240):
    states = []
    for theta_2 in np.linspace(theta_start, theta_end, angle_samples):
        states.append(
            {
                "mode": 3,
                "L1": L1,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": 0.0,
                "theta_1": theta_1,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode3_theta1_curve(csm, L1, theta1_start, theta1_end, theta_2, angle_samples=240):
    states = []
    for theta_1 in np.linspace(theta1_start, theta1_end, angle_samples):
        states.append(
            {
                "mode": 3,
                "L1": L1,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": 0.0,
                "theta_1": theta_1,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode3_l1_curve(csm, theta_2, l1_start, l1_end, length_samples=240):
    states = []
    for L1 in np.linspace(l1_start, l1_end, length_samples):
        states.append(
            {
                "mode": 3,
                "L1": L1,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": 0.0,
                "theta_1": min(csm.kappa_10 * L1, csm.theta1_limit),
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode4_theta1_curve(csm, Ls, theta1_start, theta1_end, theta_2, angle_samples=240):
    states = []
    for theta_1 in np.linspace(theta1_start, theta1_end, angle_samples):
        states.append(
            {
                "mode": 4,
                "L1": csm.L_10,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": Ls,
                "theta_1": theta_1,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode4_theta2_curve(csm, Ls, theta_1, theta_start, theta_end, angle_samples=240):
    states = []
    for theta_2 in np.linspace(theta_start, theta_end, angle_samples):
        states.append(
            {
                "mode": 4,
                "L1": csm.L_10,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": Ls,
                "theta_1": theta_1,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def _sample_mode4_ls_curve(csm, theta_1, theta_2, ls_start, ls_end, length_samples=240):
    states = []
    for Ls in np.linspace(ls_start, ls_end, length_samples):
        states.append(
            {
                "mode": 4,
                "L1": csm.L_10,
                "L2": csm.L_20,
                "Lr": csm.L_r0,
                "Ls": Ls,
                "theta_1": theta_1,
                "theta_2": theta_2,
            }
        )
    return _sample_state_curve(csm, states)


def build_mode1_profile(csm, options: BoundaryScanOptions | None = None):
    options = options or BoundaryScanOptions()
    inner_curve = _sample_mode1_inner_curve(csm, length_samples=options.length_samples)
    outer_curve = _sample_mode2_theta_sweep_curve(csm, Lr=0.0, angle_samples=options.angle_samples)
    axis_closure = _build_axis_closure(outer_curve[-1], inner_curve[0])
    closed_profile = np.vstack((inner_curve, outer_curve[1:], axis_closure[1:]))
    return WorkspaceProfile(
        mode=1,
        inner_segments=[inner_curve],
        outer_segments=[outer_curve],
        outer_open_curve_rz=outer_curve,
        unreachable_open_curves_rz=[inner_curve],
        closed_profile_rz=closed_profile,
        unreachable_closed_profiles_rz=[_close_curve_to_axis(inner_curve)],
        debug_data=None,
    )


def build_mode2_profile(csm, options: BoundaryScanOptions | None = None):
    options = options or BoundaryScanOptions()
    length_samples = options.length_samples
    angle_samples = options.angle_samples
    theta_break = min(csm.theta2_limit, 0.5 * np.pi)

    primitives = {
        "tau0": BoundaryPrimitive("tau0", _sample_mode2_theta_curve(csm, Lr=0.0, theta_start=0.0, theta_end=csm.theta2_limit, angle_samples=angle_samples)),
        "tau1": BoundaryPrimitive("tau1", _sample_mode2_theta_curve(csm, Lr=csm.L_r0, theta_start=0.0, theta_end=csm.theta2_limit, angle_samples=angle_samples)),
        "tau2": BoundaryPrimitive("tau2", _sample_mode2_lr_curve(csm, theta_2=csm.theta2_limit, lr_start=csm.L_r0, lr_end=0.0, length_samples=length_samples)),
    }

    if csm.theta2_limit > 0.5 * np.pi + 1e-9:
        outer_segments = [
            _sample_mode2_theta_curve(csm, Lr=0.0, theta_start=csm.theta2_limit, theta_end=theta_break, angle_samples=angle_samples),
            _sample_mode2_lr_curve(csm, theta_2=theta_break, lr_start=0.0, lr_end=csm.L_r0, length_samples=length_samples),
            _sample_mode2_theta_curve(csm, Lr=csm.L_r0, theta_start=theta_break, theta_end=0.0, angle_samples=angle_samples),
        ]
    else:
        outer_segments = [
            _sample_mode2_lr_curve(csm, theta_2=csm.theta2_limit, lr_start=0.0, lr_end=csm.L_r0, length_samples=length_samples),
            _sample_mode2_theta_curve(csm, Lr=csm.L_r0, theta_start=csm.theta2_limit, theta_end=0.0, angle_samples=angle_samples),
        ]

    hit_tau1 = _first_polyline_intersection(primitives["tau0"].points_rz, primitives["tau1"].points_rz)
    hit_tau2 = _first_polyline_intersection(primitives["tau0"].points_rz, primitives["tau2"].points_rz)
    candidates = []
    if hit_tau1 is not None:
        candidates.append(("tau1", hit_tau1))
    if hit_tau2 is not None:
        candidates.append(("tau2", hit_tau2))

    chosen_hit_name = None
    chosen_hit = None
    if candidates:
        hit_name, hit = min(candidates, key=lambda item: item[1]["idx_a"] + item[1]["t_a"])
        chosen_hit_name = hit_name
        chosen_hit = hit
        tau0_prefix = _polyline_prefix(primitives["tau0"].points_rz, hit)
        if hit_name == "tau1":
            tau1_suffix = _polyline_suffix(primitives["tau1"].points_rz, hit, which="b")
            inner_segments = [tau0_prefix, tau1_suffix, primitives["tau2"].points_rz]
        else:
            tau2_suffix = _polyline_suffix(primitives["tau2"].points_rz, hit, which="b")
            inner_segments = [tau0_prefix, tau2_suffix]
        inner_curve = _concat_curve_segments(inner_segments)
    else:
        inner_curve = primitives["tau0"].points_rz.copy()

    axis_connector = _build_axis_closure(inner_curve[-1], outer_segments[-1][-1])
    outer_path = _concat_curve_segments([segment[::-1] for segment in outer_segments[::-1]])
    closed_profile = _concat_curve_segments((inner_curve, axis_connector[1:], outer_path[1:]))
    return WorkspaceProfile(
        mode=2,
        inner_segments=[inner_curve],
        outer_segments=outer_segments,
        outer_open_curve_rz=_concat_curve_segments(outer_segments),
        unreachable_open_curves_rz=[inner_curve],
        closed_profile_rz=closed_profile,
        unreachable_closed_profiles_rz=[_close_curve_to_axis(inner_curve)],
        debug_data={
            "primitives": {name: primitive.points_rz.copy() for name, primitive in primitives.items()},
            "outer_segments": [segment.copy() for segment in outer_segments],
            "inner_segments": [segment.copy() for segment in inner_segments] if candidates else [inner_curve.copy()],
            "hit_tau1": None if hit_tau1 is None else dict(hit_tau1),
            "hit_tau2": None if hit_tau2 is None else dict(hit_tau2),
            "chosen_hit_name": chosen_hit_name,
            "chosen_hit": None if chosen_hit is None else dict(chosen_hit),
            "theta_break": float(theta_break),
        },
    )


def build_mode3_profile(csm, options: BoundaryScanOptions | None = None):
    options = options or BoundaryScanOptions()
    length_samples = options.length_samples
    angle_samples = options.angle_samples
    primitives = {
        "tau0": BoundaryPrimitive("tau0", _sample_mode3_theta2_curve(csm, L1=0.0, theta_1=0.0, theta_start=0.0, theta_end=csm.theta2_limit, angle_samples=angle_samples)),
        "tau1": BoundaryPrimitive("tau1", _sample_mode3_l1_curve(csm, theta_2=csm.theta2_limit, l1_start=0.0, l1_end=csm.L_10, length_samples=length_samples)),
        "tau2": BoundaryPrimitive("tau2", _sample_mode3_theta2_curve(csm, L1=csm.L_10, theta_1=csm.theta1_limit, theta_start=csm.theta2_limit, theta_end=0.0, angle_samples=angle_samples)),
        "tau3": BoundaryPrimitive("tau3", _sample_mode3_theta1_curve(csm, L1=csm.L_10, theta1_start=csm.theta1_limit, theta1_end=0.0, theta_2=0.0, angle_samples=angle_samples)),
    }
    inner_segments = [primitives["tau0"].points_rz, primitives["tau1"].points_rz]
    outer_segments = [primitives["tau2"].points_rz, primitives["tau3"].points_rz]
    inner_curve = _concat_curve_segments(inner_segments)
    axis_connector = _build_axis_closure(inner_curve[-1], outer_segments[-1][-1])
    outer_path = _concat_curve_segments([segment[::-1] for segment in outer_segments[::-1]])
    closed_profile = _concat_curve_segments((inner_curve, axis_connector[1:], outer_path[1:]))
    return WorkspaceProfile(
        mode=3,
        inner_segments=[segment.copy() for segment in inner_segments],
        outer_segments=[segment.copy() for segment in outer_segments],
        outer_open_curve_rz=_concat_curve_segments(outer_segments),
        unreachable_open_curves_rz=[inner_curve],
        closed_profile_rz=closed_profile,
        unreachable_closed_profiles_rz=[_close_curve_to_axis(inner_curve)],
        debug_data={
            "primitives": {name: primitive.points_rz.copy() for name, primitive in primitives.items()},
            "outer_segments": [segment.copy() for segment in outer_segments],
            "inner_segments": [segment.copy() for segment in inner_segments],
        },
    )


def build_mode4_profile(csm, options: BoundaryScanOptions | None = None):
    options = options or BoundaryScanOptions()
    length_samples = options.length_samples
    angle_samples = options.angle_samples
    theta1_break = min(csm.theta1_limit, 0.5 * np.pi)
    theta2_pre_curve = _sample_mode4_theta2_curve(csm, Ls=csm.L_s0, theta_1=theta1_break, theta_start=0.0, theta_end=csm.theta2_limit, angle_samples=angle_samples)
    theta2_break = 0.0
    if theta1_break < 0.5 * np.pi - 1e-9 and theta2_pre_curve.shape[0] >= 2:
        theta2_idx = int(np.argmax(theta2_pre_curve[:, 0]))
        theta2_break = float(np.linspace(0.0, csm.theta2_limit, angle_samples)[theta2_idx])

    primitives = {
        "outer_theta1": BoundaryPrimitive("outer_theta1", _sample_mode4_theta1_curve(csm, Ls=csm.L_s0, theta1_start=0.0, theta1_end=theta1_break, theta_2=0.0, angle_samples=angle_samples)),
        "outer_theta2_pre": BoundaryPrimitive("outer_theta2_pre", theta2_pre_curve),
        "outer_ls": BoundaryPrimitive("outer_ls", _sample_mode4_ls_curve(csm, theta_1=theta1_break, theta_2=theta2_break, ls_start=csm.L_s0, ls_end=0.0, length_samples=length_samples)),
        "outer_theta2": BoundaryPrimitive("outer_theta2", _sample_mode4_theta2_curve(csm, Ls=0.0, theta_1=theta1_break, theta_start=theta2_break, theta_end=csm.theta2_limit, angle_samples=angle_samples)),
        "inner_family_theta2": BoundaryPrimitive("inner_family_theta2", _sample_mode4_theta2_curve(csm, Ls=0.0, theta_1=theta1_break, theta_start=0.0, theta_end=csm.theta2_limit, angle_samples=angle_samples)),
        "mode3_inner_theta2": BoundaryPrimitive("mode3_inner_theta2", _sample_mode3_theta2_curve(csm, L1=0.0, theta_1=0.0, theta_start=0.0, theta_end=csm.theta2_limit, angle_samples=angle_samples)),
        "mode3_inner_l1": BoundaryPrimitive("mode3_inner_l1", _sample_mode3_l1_curve(csm, theta_2=csm.theta2_limit, l1_start=0.0, l1_end=csm.L_10, length_samples=length_samples)),
        "inner_ls_cover": BoundaryPrimitive("inner_ls_cover", _sample_mode4_ls_curve(csm, theta_1=csm.theta1_limit, theta_2=csm.theta2_limit, ls_start=0.0, ls_end=csm.L_s0, length_samples=length_samples)),
    }

    outer_segments = [primitives["outer_theta1"].points_rz]
    if theta2_break > 1e-9:
        outer_segments.append(_sample_mode4_theta2_curve(csm, Ls=csm.L_s0, theta_1=theta1_break, theta_start=0.0, theta_end=theta2_break, angle_samples=angle_samples))
    outer_segments.extend([primitives["outer_ls"].points_rz, primitives["outer_theta2"].points_rz])

    base_inner_seg1 = primitives["mode3_inner_theta2"].points_rz
    base_inner_seg2 = primitives["mode3_inner_l1"].points_rz
    base_inner_segments = [base_inner_seg1, base_inner_seg2]

    inner_segments = [segment.copy() for segment in base_inner_segments]
    chosen_inner_mode = "mode3_inner"
    family_path = _concat_curve_segments([primitives["inner_family_theta2"].points_rz, primitives["inner_ls_cover"].points_rz[1:]])
    cover_hit_seg = None
    cover_hit = None

    hit_candidates = []
    if family_path.shape[0] >= 2:
        hit_seg1 = _first_polyline_intersection(base_inner_seg1, family_path)
        hit_seg2 = _first_polyline_intersection(base_inner_seg2, family_path)
        if hit_seg1 is not None:
            hit_candidates.append(("mode3_inner_theta2", hit_seg1))
        if hit_seg2 is not None:
            hit_candidates.append(("mode3_inner_l1", hit_seg2))

    if hit_candidates:
        cover_hit_seg, cover_hit = min(hit_candidates, key=lambda item: item[1]["idx_a"] + item[1]["t_a"])
        family_suffix = _polyline_suffix(family_path, cover_hit, which="b")
        if cover_hit_seg == "mode3_inner_theta2":
            inner_segments = [_polyline_prefix(base_inner_seg1, cover_hit), family_suffix]
        else:
            inner_segments = [base_inner_seg1.copy(), _polyline_prefix(base_inner_seg2, cover_hit), family_suffix]
        chosen_inner_mode = f"family_path_trims_{cover_hit_seg}"

    inner_curve = _concat_curve_segments(inner_segments)
    axis_connector = _build_axis_closure(inner_curve[-1], outer_segments[-1][-1])
    outer_path = _concat_curve_segments([segment[::-1] for segment in outer_segments[::-1]])
    closed_profile = _concat_curve_segments((inner_curve, axis_connector[1:], outer_path[1:]))
    return WorkspaceProfile(
        mode=4,
        inner_segments=[segment.copy() for segment in inner_segments],
        outer_segments=[segment.copy() for segment in outer_segments],
        outer_open_curve_rz=_concat_curve_segments(outer_segments),
        unreachable_open_curves_rz=[inner_curve],
        closed_profile_rz=closed_profile,
        unreachable_closed_profiles_rz=[_close_curve_to_axis(inner_curve)],
        debug_data={
            "primitives": {name: primitive.points_rz.copy() for name, primitive in primitives.items()},
            "outer_segments": [segment.copy() for segment in outer_segments],
            "inner_segments": [segment.copy() for segment in inner_segments],
            "theta1_break": float(theta1_break),
            "theta2_break": float(theta2_break),
            "family_path": family_path.copy(),
            "cover_hit_seg": cover_hit_seg,
            "cover_hit": None if cover_hit is None else dict(cover_hit),
            "chosen_inner_mode": chosen_inner_mode,
        },
    )


def build_mode0_profile(csm, options: BoundaryScanOptions | None = None):
    options = options or BoundaryScanOptions()
    route_options = BoundaryScanOptions(
        length_samples=min(options.length_samples, options.mode0_route_length_samples),
        angle_samples=min(options.angle_samples, options.mode0_route_angle_samples),
        mode0_node_merge_tol=options.mode0_node_merge_tol,
        mode0_endpoint_snap_tol=options.mode0_endpoint_snap_tol,
        mode0_route_length_samples=options.mode0_route_length_samples,
        mode0_route_angle_samples=options.mode0_route_angle_samples,
        mode0_debug_output_dir=options.mode0_debug_output_dir,
    )
    source_profiles = _collect_mode0_sources(csm, route_options)
    outer_source_mode = 4 if source_profiles.get(4) is not None else 3
    outer_profile = source_profiles[outer_source_mode]

    curves = []
    for mode in sorted(source_profiles):
        profile = source_profiles[mode]
        if profile is None:
            continue
        for idx, segment in enumerate(profile.inner_segments):
            curves.append({"name": f"mode{mode}_inner_{idx}", "role": "inner", "points": np.asarray(segment, dtype=float)})
        for idx, segment in enumerate(profile.outer_segments):
            curves.append({"name": f"mode{mode}_outer_{idx}", "role": "outer", "points": np.asarray(segment, dtype=float)})

    start_curve_name = "mode1_inner_0"
    target_curve_names = [f"mode{outer_source_mode}_outer_{idx}" for idx in range(len(outer_profile.outer_segments))]
    try:
        inner_path_forward, trace_debug = _trace_right_turn_path(
            curves,
            start_curve_name,
            target_curve_names,
            node_merge_tol=options.mode0_node_merge_tol,
            endpoint_snap_tol=options.mode0_endpoint_snap_tol,
        )
    except RuntimeError as exc:
        if options.mode0_debug_output_dir is not None:
            report_path = Path(options.mode0_debug_output_dir) / "mode0_trace_failure.txt"
            _write_mode0_debug_report(curves, str(exc), report_path)
            raise RuntimeError(f"{exc} Debug report written to: {report_path.resolve()}") from exc
        raise

    inner_curve = inner_path_forward.copy()
    all_points = [outer_profile.outer_open_curve_rz]
    all_points.extend(curve["points"] for curve in curves)
    min_z = float(np.min(np.concatenate([points[:, 1] for points in all_points if len(points) > 0])))
    inner_curve = _extend_curve_outer_endpoint_downward(inner_curve, min_z)

    axis_connector = _build_axis_closure(inner_curve[-1], outer_profile.outer_segments[-1][-1])
    outer_path = _concat_curve_segments([segment[::-1] for segment in outer_profile.outer_segments[::-1]])
    closed_profile = _concat_curve_segments((inner_curve, axis_connector[1:], outer_path[1:]))
    return WorkspaceProfile(
        mode=0,
        inner_segments=[inner_curve],
        outer_segments=[segment.copy() for segment in outer_profile.outer_segments],
        outer_open_curve_rz=outer_profile.outer_open_curve_rz.copy(),
        unreachable_open_curves_rz=[inner_curve],
        closed_profile_rz=closed_profile,
        unreachable_closed_profiles_rz=[_close_curve_to_axis(inner_curve)],
        debug_data={
            "outer_source_mode": int(outer_source_mode),
            "source_profiles": [mode for mode, profile in source_profiles.items() if profile is not None],
            "route_length_samples": int(route_options.length_samples),
            "route_angle_samples": int(route_options.angle_samples),
            "outer_segments": [segment.copy() for segment in outer_profile.outer_segments],
            "inner_segments": [inner_curve.copy()],
            "trace_path": inner_path_forward.copy(),
            "trace_segments": [segment.copy() for segment in trace_debug["path_segments"]],
            "trace_start_node": trace_debug["start_node"],
            "trace_target_nodes": trace_debug["target_nodes"],
            "all_curves": [(curve["name"], curve["role"], curve["points"].copy()) for curve in curves],
        },
    )


def build_workspace_profile(csm, mode, options: BoundaryScanOptions | None = None):
    options = options or BoundaryScanOptions()
    if mode == 0:
        return build_mode0_profile(csm, options)
    if mode == 1:
        return build_mode1_profile(csm, options)
    if mode == 2:
        return build_mode2_profile(csm, options)
    if mode == 3:
        return build_mode3_profile(csm, options)
    if mode == 4:
        return build_mode4_profile(csm, options)
    raise NotImplementedError(f"Mode {mode} is not implemented yet in boundary-scan plotting.")


def build_workspace_profiles(csm, modes, options: BoundaryScanOptions | None = None):
    options = options or BoundaryScanOptions()
    return [build_workspace_profile(csm, mode, options) for mode in modes]


def _state_dict(*, mode, L1, L2, Lr, Ls, theta_1, theta_2, phi=0.0, delta_1=0.0, delta_2=0.0):
    return {
        "mode": int(mode),
        "phi": float(phi),
        "L1": float(L1),
        "L2": float(L2),
        "Lr": float(Lr),
        "Ls": float(Ls),
        "theta_1": float(theta_1),
        "theta_2": float(theta_2),
        "delta_1": float(delta_1),
        "delta_2": float(delta_2),
    }


def _copy_state_sequence(states):
    return [dict(state) for state in states]


def _extract_csm_spec(csm):
    return {
        "L_10": float(csm.L_10),
        "L_20": float(csm.L_20),
        "L_r0": float(csm.L_r0),
        "L_s0": float(csm.L_s0),
        "L_tool": float(csm.L_tool),
        "theta1_max": float(csm.theta1_max),
        "theta2_max": float(csm.theta2_max),
        "delta_t": float(csm.delta_t),
        "r1_min": float(csm.r1_min),
        "r2_min": float(csm.r2_min),
    }


def _copy_animation_curve(points_rz, **kwargs):
    return WorkspaceAnimationCurve(points_rz=np.asarray(points_rz, dtype=float).copy(), **kwargs)


def _animation_stage(title, curves, annotation=None, frames=18, state_sequence=None):
    return WorkspaceAnimationStage(
        title=title,
        curves=list(curves),
        annotation=annotation,
        frames=frames,
        state_sequence=[] if state_sequence is None else _copy_state_sequence(state_sequence),
    )


def _mode1_inner_states(csm, length_samples):
    return [
        _state_dict(
            mode=1,
            L1=0.0,
            L2=L2,
            Lr=0.0,
            Ls=0.0,
            theta_1=0.0,
            theta_2=min(csm.kappa_20 * L2, csm.theta2_limit),
        )
        for L2 in np.linspace(0.0, csm.L_20, length_samples)
    ]


def _mode2_theta_states(csm, Lr, theta_start, theta_end, angle_samples):
    return [
        _state_dict(
            mode=2,
            L1=0.0,
            L2=csm.L_20,
            Lr=Lr,
            Ls=0.0,
            theta_1=0.0,
            theta_2=theta_2,
        )
        for theta_2 in np.linspace(theta_start, theta_end, angle_samples)
    ]


def _mode2_lr_states(csm, theta_2, lr_start, lr_end, length_samples):
    return [
        _state_dict(
            mode=2,
            L1=0.0,
            L2=csm.L_20,
            Lr=Lr,
            Ls=0.0,
            theta_1=0.0,
            theta_2=theta_2,
        )
        for Lr in np.linspace(lr_start, lr_end, length_samples)
    ]


def _mode3_theta2_states(csm, L1, theta_1, theta_start, theta_end, angle_samples):
    return [
        _state_dict(
            mode=3,
            L1=L1,
            L2=csm.L_20,
            Lr=csm.L_r0,
            Ls=0.0,
            theta_1=theta_1,
            theta_2=theta_2,
        )
        for theta_2 in np.linspace(theta_start, theta_end, angle_samples)
    ]


def _mode3_l1_states(csm, theta_2, l1_start, l1_end, length_samples):
    return [
        _state_dict(
            mode=3,
            L1=L1,
            L2=csm.L_20,
            Lr=csm.L_r0,
            Ls=0.0,
            theta_1=min(csm.kappa_10 * L1, csm.theta1_limit),
            theta_2=theta_2,
        )
        for L1 in np.linspace(l1_start, l1_end, length_samples)
    ]


def _mode3_theta1_states(csm, L1, theta1_start, theta1_end, theta_2, angle_samples):
    return [
        _state_dict(
            mode=3,
            L1=L1,
            L2=csm.L_20,
            Lr=csm.L_r0,
            Ls=0.0,
            theta_1=theta_1,
            theta_2=theta_2,
        )
        for theta_1 in np.linspace(theta1_start, theta1_end, angle_samples)
    ]


def _mode4_theta1_states(csm, Ls, theta1_start, theta1_end, theta_2, angle_samples):
    return [
        _state_dict(
            mode=4,
            L1=csm.L_10,
            L2=csm.L_20,
            Lr=csm.L_r0,
            Ls=Ls,
            theta_1=theta_1,
            theta_2=theta_2,
        )
        for theta_1 in np.linspace(theta1_start, theta1_end, angle_samples)
    ]


def _mode4_theta2_states(csm, Ls, theta_1, theta_start, theta_end, angle_samples):
    return [
        _state_dict(
            mode=4,
            L1=csm.L_10,
            L2=csm.L_20,
            Lr=csm.L_r0,
            Ls=Ls,
            theta_1=theta_1,
            theta_2=theta_2,
        )
        for theta_2 in np.linspace(theta_start, theta_end, angle_samples)
    ]


def _mode4_ls_states(csm, theta_1, theta_2, ls_start, ls_end, length_samples):
    return [
        _state_dict(
            mode=4,
            L1=csm.L_10,
            L2=csm.L_20,
            Lr=csm.L_r0,
            Ls=Ls,
            theta_1=theta_1,
            theta_2=theta_2,
        )
        for Ls in np.linspace(ls_start, ls_end, length_samples)
    ]


def _build_mode1_animation_data(profile: WorkspaceProfile, csm=None):
    inner_curve = profile.inner_segments[0]
    outer_curve = profile.outer_segments[0]
    inner_states = [] if csm is None else _mode1_inner_states(csm, max(len(inner_curve), 2))
    outer_states = [] if csm is None else _mode2_theta_states(csm, 0.0, csm.theta2_limit, 0.0, max(len(outer_curve), 2))
    stages = [
        _animation_stage(
            "Mode 1: Inner Boundary",
            [_copy_animation_curve(inner_curve, label="Inner contour", color="#D79A6B")],
            annotation="Sweep the insertion length to form the inner forbidden boundary.",
            state_sequence=inner_states,
        ),
        _animation_stage(
            "Mode 1: Outer Boundary",
            [
                _copy_animation_curve(inner_curve, label="Inner contour", color="#D79A6B", progressive=False, alpha=0.55),
                _copy_animation_curve(outer_curve, label="Outer contour", color="#2D5F73"),
            ],
            annotation="Sweep the outer family to close the reachable shell.",
            state_sequence=outer_states,
        ),
        _animation_stage(
            "Mode 1: Final Profile",
            [
                _copy_animation_curve(inner_curve, label="Inner contour", color="#D79A6B", progressive=False),
                _copy_animation_curve(outer_curve, label="Outer contour", color="#2D5F73", progressive=False),
            ],
            annotation="Final side-view boundary for mode 1.",
            frames=12,
            state_sequence=[] if not outer_states else [outer_states[-1]],
        ),
    ]
    return WorkspaceAnimationData(mode=profile.mode, stages=stages, csm_spec=None if csm is None else _extract_csm_spec(csm))


def _build_mode2_animation_data(profile: WorkspaceProfile, csm=None):
    debug = profile.debug_data or {}
    primitives = debug.get("primitives", {})
    inner_segments = debug.get("inner_segments", profile.inner_segments)
    outer_segments = debug.get("outer_segments", profile.outer_segments)
    tau0_states = [] if csm is None else _mode2_theta_states(csm, 0.0, 0.0, csm.theta2_limit, max(len(primitives.get("tau0", [])), 2))
    tau1_states = [] if csm is None else _mode2_theta_states(csm, csm.L_r0, 0.0, csm.theta2_limit, max(len(primitives.get("tau1", [])), 2))
    tau2_states = [] if csm is None else _mode2_lr_states(csm, csm.theta2_limit, csm.L_r0, 0.0, max(len(primitives.get("tau2", [])), 2))
    stages = [
        _animation_stage(
            "Mode 2: tau0",
            [_copy_animation_curve(primitives["tau0"], label="tau0", color="#C96868")] if "tau0" in primitives else [],
            annotation="Trace the base family tau0.",
            state_sequence=tau0_states,
        ),
        _animation_stage(
            "Mode 2: tau1",
            [
                _copy_animation_curve(primitives["tau0"], label="tau0", color="#C96868", progressive=False, alpha=0.65),
                _copy_animation_curve(primitives["tau1"], label="tau1", color="#7C5CFC") if "tau1" in primitives else None,
            ],
            annotation="Trace tau1 while keeping tau0 as reference.",
            state_sequence=tau1_states,
        ),
        _animation_stage(
            "Mode 2: tau2",
            [
                _copy_animation_curve(primitives["tau0"], label="tau0", color="#C96868", progressive=False, alpha=0.65),
                _copy_animation_curve(primitives["tau1"], label="tau1", color="#7C5CFC", progressive=False, alpha=0.65) if "tau1" in primitives else None,
                _copy_animation_curve(primitives["tau2"], label="tau2", color="#2E8B57") if "tau2" in primitives else None,
            ],
            annotation="Build the candidate families used to trim the inner boundary.",
            state_sequence=tau2_states,
        ),
    ]
    for stage in stages:
        if stage.curves:
            stage.curves = [curve for curve in stage.curves if curve is not None]

    hit_curves = []
    for hit_key, color, label in (
        ("hit_tau1", "#7C5CFC", "hit tau1"),
        ("hit_tau2", "#2E8B57", "hit tau2"),
        ("chosen_hit", "#111111", "chosen hit"),
    ):
        hit = debug.get(hit_key)
        if hit is not None:
            hit_curves.append(
                _copy_animation_curve(
                    np.asarray(hit["point"], dtype=float)[np.newaxis, :],
                    label=label,
                    color=color,
                    progressive=False,
                    marker="o",
                    linewidth=0.0,
                    mirror=True,
                )
            )
    stages.append(
        _animation_stage(
            "Mode 2: Intersection Selection",
            [curve for curve in stages[-1].curves] + hit_curves,
            annotation="Pick the earliest intersection to define the trimmed inner boundary.",
            state_sequence=[] if not tau2_states else [tau2_states[-1]],
        )
    )
    stages.append(
        _animation_stage(
            "Mode 2: Final Boundary",
            [
                *[
                    _copy_animation_curve(segment, label=f"inner_{idx}", color="#D79A6B", progressive=False)
                    for idx, segment in enumerate(inner_segments)
                ],
                *[
                    _copy_animation_curve(segment, label=f"outer_{idx}", color="#2D5F73", progressive=False)
                    for idx, segment in enumerate(outer_segments)
                ],
            ],
            annotation="Compose the final inner and outer contours.",
            frames=22,
            state_sequence=[] if not tau2_states else [tau2_states[-1]],
        )
    )
    return WorkspaceAnimationData(mode=profile.mode, stages=stages, csm_spec=None if csm is None else _extract_csm_spec(csm))


def _build_mode3_animation_data(profile: WorkspaceProfile, csm=None):
    debug = profile.debug_data or {}
    primitives = debug.get("primitives", {})
    primitive_colors = {
        "tau0": "#C96868",
        "tau1": "#2E8B57",
        "tau2": "#7C5CFC",
        "tau3": "#1F5D78",
    }
    stages = []
    visible = []
    state_builders = {}
    if csm is not None:
        state_builders = {
            "tau0": _mode3_theta2_states(csm, 0.0, 0.0, 0.0, csm.theta2_limit, max(len(primitives.get("tau0", [])), 2)),
            "tau1": _mode3_l1_states(csm, csm.theta2_limit, 0.0, csm.L_10, max(len(primitives.get("tau1", [])), 2)),
            "tau2": _mode3_theta2_states(csm, csm.L_10, csm.theta1_limit, csm.theta2_limit, 0.0, max(len(primitives.get("tau2", [])), 2)),
            "tau3": _mode3_theta1_states(csm, csm.L_10, csm.theta1_limit, 0.0, 0.0, max(len(primitives.get("tau3", [])), 2)),
        }
    for name in ("tau0", "tau1", "tau2", "tau3"):
        curve = primitives.get(name)
        if curve is None:
            continue
        visible = [*visible, _copy_animation_curve(curve, label=name, color=primitive_colors[name])]
        stages.append(
            _animation_stage(
                f"Mode 3: {name}",
                [
                    _copy_animation_curve(item.points_rz, label=item.label, color=item.color, progressive=False if idx < len(visible) - 1 else True)
                    for idx, item in enumerate(visible)
                ],
                annotation="Accumulate the primitive families that define the mode 3 boundary.",
                state_sequence=state_builders.get(name, []),
            )
        )
    stages.append(
        _animation_stage(
            "Mode 3: Final Boundary",
            [
                *[
                    _copy_animation_curve(segment, label=f"inner_{idx}", color="#D79A6B", progressive=False)
                    for idx, segment in enumerate(profile.inner_segments)
                ],
                *[
                    _copy_animation_curve(segment, label=f"outer_{idx}", color="#2D5F73", progressive=False)
                    for idx, segment in enumerate(profile.outer_segments)
                ],
            ],
            annotation="Combine the primitive families into the final profile.",
            frames=22,
            state_sequence=[] if "tau3" not in state_builders else [state_builders["tau3"][-1]],
        )
    )
    return WorkspaceAnimationData(mode=profile.mode, stages=stages, csm_spec=None if csm is None else _extract_csm_spec(csm))


def _build_mode4_animation_data(profile: WorkspaceProfile, csm=None):
    debug = profile.debug_data or {}
    primitives = debug.get("primitives", {})
    family_path = debug.get("family_path")
    outer_theta1_states = [] if csm is None else _mode4_theta1_states(csm, csm.L_s0, 0.0, debug.get("theta1_break", csm.theta1_limit), 0.0, max(len(primitives.get("outer_theta1", [])), 2))
    outer_theta2_pre_states = [] if csm is None else _mode4_theta2_states(csm, csm.L_s0, debug.get("theta1_break", csm.theta1_limit), 0.0, csm.theta2_limit, max(len(primitives.get("outer_theta2_pre", [])), 2))
    outer_ls_states = [] if csm is None else _mode4_ls_states(csm, debug.get("theta1_break", csm.theta1_limit), debug.get("theta2_break", 0.0), csm.L_s0, 0.0, max(len(primitives.get("outer_ls", [])), 2))
    outer_theta2_states = [] if csm is None else _mode4_theta2_states(csm, 0.0, debug.get("theta1_break", csm.theta1_limit), debug.get("theta2_break", 0.0), csm.theta2_limit, max(len(primitives.get("outer_theta2", [])), 2))
    inner_family_states = [] if csm is None else _mode4_theta2_states(csm, 0.0, debug.get("theta1_break", csm.theta1_limit), 0.0, csm.theta2_limit, max(len(primitives.get("inner_family_theta2", [])), 2))
    stages = [
        _animation_stage(
            "Mode 4: outer_theta1",
            [_copy_animation_curve(primitives["outer_theta1"], label="outer_theta1", color="#2D5F73")] if "outer_theta1" in primitives else [],
            annotation="Build the first outer family branch for mode 4.",
            frames=22,
            state_sequence=outer_theta1_states,
        ),
        _animation_stage(
            "Mode 4: outer_theta2_pre",
            [
                _copy_animation_curve(primitives["outer_theta1"], label="outer_theta1", color="#2D5F73", progressive=False, alpha=0.65) if "outer_theta1" in primitives else None,
                _copy_animation_curve(primitives["outer_theta2_pre"], label="outer_theta2_pre", color="#2D5F73") if "outer_theta2_pre" in primitives else None,
            ],
            annotation="Continue the outer family with the pre-break theta2 sweep.",
            frames=22,
            state_sequence=outer_theta2_pre_states,
        ),
        _animation_stage(
            "Mode 4: outer_ls",
            [
                _copy_animation_curve(primitives["outer_theta1"], label="outer_theta1", color="#2D5F73", progressive=False, alpha=0.55) if "outer_theta1" in primitives else None,
                _copy_animation_curve(primitives["outer_theta2_pre"], label="outer_theta2_pre", color="#2D5F73", progressive=False, alpha=0.55) if "outer_theta2_pre" in primitives else None,
                _copy_animation_curve(primitives["outer_ls"], label="outer_ls", color="#2D5F73") if "outer_ls" in primitives else None,
            ],
            annotation="Sweep the base length branch of the outer family.",
            frames=22,
            state_sequence=outer_ls_states,
        ),
        _animation_stage(
            "Mode 4: outer_theta2",
            [
                _copy_animation_curve(primitives["outer_theta1"], label="outer_theta1", color="#2D5F73", progressive=False, alpha=0.45) if "outer_theta1" in primitives else None,
                _copy_animation_curve(primitives["outer_theta2_pre"], label="outer_theta2_pre", color="#2D5F73", progressive=False, alpha=0.45) if "outer_theta2_pre" in primitives else None,
                _copy_animation_curve(primitives["outer_ls"], label="outer_ls", color="#2D5F73", progressive=False, alpha=0.45) if "outer_ls" in primitives else None,
                _copy_animation_curve(primitives["outer_theta2"], label="outer_theta2", color="#2D5F73") if "outer_theta2" in primitives else None,
            ],
            annotation="Finish the outer family with the final theta2 sweep.",
            frames=22,
            state_sequence=outer_theta2_states,
        ),
        _animation_stage(
            "Mode 4: Inner Family",
            [
                *[
                    _copy_animation_curve(primitives[name], label=name, color="#D79A6B")
                    for name in ("mode3_inner_theta2", "mode3_inner_l1", "inner_family_theta2", "inner_ls_cover")
                    if name in primitives
                ],
                _copy_animation_curve(family_path, label="family_path", color="#111111", linestyle="--")
                if family_path is not None else None,
            ],
            annotation="Construct the candidate inner family used for cover trimming.",
            frames=22,
            state_sequence=inner_family_states,
        ),
    ]
    for stage in stages:
        if stage.curves:
            stage.curves = [curve for curve in stage.curves if curve is not None]

    hit = debug.get("cover_hit")
    if hit is not None:
        stages.append(
            _animation_stage(
                "Mode 4: Cover Hit",
                [
                    *[
                        _copy_animation_curve(curve.points_rz, label=curve.label, color=curve.color, progressive=False, linestyle=curve.linestyle)
                        for curve in stages[-1].curves
                    ],
                    _copy_animation_curve(
                        np.asarray(hit["point"], dtype=float)[np.newaxis, :],
                        label="cover_hit",
                        color="#111111",
                        progressive=False,
                        marker="o",
                        linewidth=0.0,
                    ),
                ],
                annotation="Locate the cover-family intersection used to trim the inner contour.",
                state_sequence=[] if not inner_family_states else [inner_family_states[-1]],
            )
        )

    stages.append(
        _animation_stage(
            "Mode 4: Final Boundary",
            [
                *[
                    _copy_animation_curve(segment, label=f"inner_{idx}", color="#D79A6B", progressive=False)
                    for idx, segment in enumerate(profile.inner_segments)
                ],
                *[
                    _copy_animation_curve(segment, label=f"outer_{idx}", color="#2D5F73", progressive=False)
                    for idx, segment in enumerate(profile.outer_segments)
                ],
            ],
            annotation="Compose the trimmed inner boundary and the final outer shell.",
            frames=24,
            state_sequence=[] if not outer_theta2_states else [outer_theta2_states[-1]],
        )
    )
    return WorkspaceAnimationData(mode=profile.mode, stages=stages, csm_spec=None if csm is None else _extract_csm_spec(csm))


def _build_mode0_animation_data(profile: WorkspaceProfile, csm=None):
    debug = profile.debug_data or {}
    network_curves = [
        _copy_animation_curve(curve, label=name, color="#7A7A7A" if role == "outer" else "#C28C62", linestyle="--", linewidth=1.4, alpha=0.7)
        for name, role, curve in debug.get("all_curves", [])
    ]
    trace_segments = [
        _copy_animation_curve(segment, label=f"trace_{idx}", color="#111111", linewidth=2.8)
        for idx, segment in enumerate(debug.get("trace_segments", []))
    ]
    stages = [
        _animation_stage(
            "Mode 0: Source Network",
            network_curves,
            annotation="Assemble all source inner and outer contours into the tracing graph.",
            frames=24,
        ),
    ]
    if trace_segments:
        stages.append(
            _animation_stage(
                "Mode 0: Right-Turn Trace",
                [
                    *[
                        _copy_animation_curve(curve.points_rz, label=curve.label, color=curve.color, linestyle=curve.linestyle, linewidth=curve.linewidth, alpha=0.35, progressive=False)
                        for curve in network_curves
                    ],
                    *trace_segments,
                ],
                annotation="Follow the right-turn rule from the mode 1 inner contour to the outer shell.",
                frames=24,
            )
        )
    stages.append(
        _animation_stage(
            "Mode 0: Final Boundary",
            [
                *[
                    _copy_animation_curve(curve.points_rz, label=curve.label, color=curve.color, linestyle=curve.linestyle, linewidth=curve.linewidth, alpha=0.25, progressive=False)
                    for curve in network_curves
                ],
                _copy_animation_curve(profile.inner_segments[0], label="final_inner", color="#111111", linewidth=3.0, progressive=False),
                *[
                    _copy_animation_curve(segment, label=f"outer_{idx}", color="#2D5F73", progressive=False)
                    for idx, segment in enumerate(profile.outer_segments)
                ],
            ],
            annotation="Promote the traced path into the final mode 0 inner boundary.",
            frames=26,
        )
    )
    return WorkspaceAnimationData(mode=profile.mode, stages=stages, csm_spec=None if csm is None else _extract_csm_spec(csm))


def build_workspace_profile_animation_data(profile: WorkspaceProfile, csm=None):
    if profile.mode == 0:
        return _build_mode0_animation_data(profile, csm=csm)
    if profile.mode == 1:
        return _build_mode1_animation_data(profile, csm=csm)
    if profile.mode == 2:
        return _build_mode2_animation_data(profile, csm=csm)
    if profile.mode == 3:
        return _build_mode3_animation_data(profile, csm=csm)
    if profile.mode == 4:
        return _build_mode4_animation_data(profile, csm=csm)
    raise NotImplementedError(f"Mode {profile.mode} is not implemented yet in boundary-scan animation.")


def build_workspace_animation_data(csm, modes, options: BoundaryScanOptions | None = None):
    profiles = build_workspace_profiles(csm, modes, options)
    return [build_workspace_profile_animation_data(profile, csm=csm) for profile in profiles]
