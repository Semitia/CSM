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
