"""GIF export with hand-colored goal points and contact borders."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from fom.preprocessing.nymeria.contacts.detection import ContactEvent, ContactInterval
from fom.preprocessing.nymeria.contacts.geometry import closest_point_on_obb
from fom.preprocessing.nymeria.contacts.validation import is_joint_contact
from fom.preprocessing.nymeria.goals.ego_projection import AriaFisheyeProjector
from fom.preprocessing.nymeria.goals.interaction_categories import HAND_COLORS, CategoryClip
from fom.preprocessing.nymeria.goals.sequence_export import rotate_image_uv_cw90
from fom.preprocessing.nymeria.io.aria_slam import RgbCameraCalibration
from fom.preprocessing.nymeria.io.objects import ObjectBBox3D

GoalMode = Literal["raw", "joint"]

BORDER_WIDTH = 8
GOAL_RADIUS = 11
NEAREST_EVENT_RADIUS = 8


def load_ordered_rgb(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    rgb = np.asarray(data["rgb"], dtype=np.uint8)
    if "frame_indices" in data.files:
        order = np.argsort(data["frame_indices"].astype(np.int64))
        rgb = rgb[order]
    return rgb


def resize_rgb(rgb: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    if max_edge <= 0:
        return rgb, 1.0
    h, w = rgb.shape[1:3]
    edge = max(h, w)
    if edge <= max_edge:
        return rgb, 1.0
    scale = max_edge / float(edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    out = np.empty((rgb.shape[0], new_h, new_w, 3), dtype=np.uint8)
    for i in range(rgb.shape[0]):
        out[i] = np.asarray(
            Image.fromarray(rgb[i]).resize((new_w, new_h), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    return out, scale


def index_events_by_frame(events: list[ContactEvent]) -> dict[int, list[ContactEvent]]:
    out: dict[int, list[ContactEvent]] = defaultdict(list)
    for ev in events:
        out[int(ev.frame_idx)].append(ev)
    return dict(out)


def _clip_object_uids(clip: CategoryClip) -> set[int]:
    if clip.chain_object_uids:
        return {int(u) for u in clip.chain_object_uids}
    if clip.object_uid >= 0:
        return {int(clip.object_uid)}
    return set()


@dataclass(frozen=True)
class ChainSegment:
    object_uid: int
    hand: str
    approach_start: int
    contact_start: int
    contact_end: int


def build_chain_segments(
    clip: CategoryClip,
    intervals: list[ContactInterval] | None,
) -> list[ChainSegment]:
    """Ordered approach/contact windows for each object in a cat4 chain."""
    uids = tuple(int(u) for u in clip.chain_object_uids)
    if len(uids) < 2:
        return []

    starts = tuple(int(s) for s in clip.chain_contact_starts)
    hands = tuple(str(h) for h in clip.chain_hands) if clip.chain_hands else ()
    if len(starts) != len(uids):
        starts = _infer_chain_contact_starts(clip, intervals, uids)

    clip_ivs = [
        iv
        for iv in (intervals or [])
        if clip.start_frame_idx <= int(iv.end_frame_idx)
        and int(iv.start_frame_idx) <= clip.end_frame_idx
    ]

    segments: list[ChainSegment] = []
    prev_contact_end = clip.start_frame_idx - 1
    for i, uid in enumerate(uids):
        c_start = int(starts[i]) if i < len(starts) else int(clip.contact_start_frame_idx)
        hand = hands[i] if i < len(hands) else str(clip.contact_hand)

        matching = [
            iv
            for iv in clip_ivs
            if int(iv.object_uid) == int(uid)
            and int(iv.start_frame_idx) <= c_start <= int(iv.end_frame_idx)
        ]
        if not matching:
            matching = [
                iv
                for iv in clip_ivs
                if int(iv.object_uid) == int(uid)
                and abs(int(iv.start_frame_idx) - c_start) <= 8
            ]
        if matching:
            iv = min(matching, key=lambda x: abs(int(x.start_frame_idx) - c_start))
            c_start = int(iv.start_frame_idx)
            c_end = int(iv.end_frame_idx)
        else:
            c_end = c_start

        approach_start = clip.start_frame_idx if i == 0 else max(clip.start_frame_idx, prev_contact_end + 1)
        segments.append(
            ChainSegment(
                object_uid=int(uid),
                hand=hand,
                approach_start=int(approach_start),
                contact_start=int(c_start),
                contact_end=int(c_end),
            )
        )
        prev_contact_end = c_end
    return segments


def _infer_chain_contact_starts(
    clip: CategoryClip,
    intervals: list[ContactInterval] | None,
    uids: tuple[int, ...],
) -> tuple[int, ...]:
    """Recover per-object contact onset when metadata omits chain_contact_starts."""
    clip_ivs = [
        iv
        for iv in (intervals or [])
        if clip.start_frame_idx <= int(iv.end_frame_idx)
        and int(iv.start_frame_idx) <= clip.end_frame_idx
    ]
    starts: list[int] = []
    cursor = clip.start_frame_idx
    for uid in uids:
        candidates = [
            iv
            for iv in clip_ivs
            if int(iv.object_uid) == int(uid) and int(iv.start_frame_idx) >= cursor
        ]
        if candidates:
            iv = min(candidates, key=lambda x: int(x.start_frame_idx))
            starts.append(int(iv.start_frame_idx))
            cursor = int(iv.end_frame_idx) + 1
        elif not starts:
            starts.append(int(clip.contact_start_frame_idx))
        else:
            starts.append(starts[-1])
    return tuple(starts)


def chain_segment_at_frame(
    clip: CategoryClip,
    global_frame: int,
    intervals: list[ContactInterval] | None,
) -> ChainSegment | None:
    """Return the chain object whose goal should be shown on ``global_frame``."""
    if clip.category != "cat4_multi_object_chain" or len(clip.chain_object_uids) < 2:
        return None

    segments = build_chain_segments(clip, intervals)
    if not segments:
        return None

    gf = int(global_frame)
    for seg in segments:
        if gf < seg.contact_start:
            if gf >= seg.approach_start:
                return seg
            return None
        if gf <= seg.contact_end:
            return seg
    return segments[-1]


@dataclass(frozen=True)
class FrameGoalState:
    in_contact: bool
    contact_hands: frozenset[str]
    goal_pairs: tuple[tuple[str, int], ...]
    chain_target_uid: int | None = None


def resolve_frame_goal_state(
    clip: CategoryClip,
    global_frame: int,
    *,
    intervals: list[ContactInterval] | None,
    events_by_frame: dict[int, list[ContactEvent]],
    goal_mode: GoalMode,
) -> FrameGoalState | None:
    """Cat4 chain-aware goal/contact state, or None for the single-object path."""
    seg = chain_segment_at_frame(clip, global_frame, intervals)
    if seg is None:
        return None

    in_contact = int(seg.contact_start) <= int(global_frame) <= int(seg.contact_end)
    contact_hands: set[str] = set()
    if in_contact:
        if goal_mode == "joint":
            active = active_hands_from_events(
                events_by_frame,
                global_frame,
                {int(seg.object_uid)},
                goal_mode="joint",
            )
            if seg.hand in active:
                contact_hands.add(seg.hand)
        else:
            contact_hands.add(seg.hand)
    return FrameGoalState(
        in_contact=in_contact,
        contact_hands=frozenset(contact_hands),
        goal_pairs=((seg.hand, int(seg.object_uid)),),
        chain_target_uid=int(seg.object_uid),
    )


def _goal_from_event(ev: ContactEvent) -> np.ndarray:
    return np.array(
        [ev.object_point_x, ev.object_point_y, ev.object_point_z],
        dtype=np.float64,
    )


def _event_matches_mode(ev: ContactEvent, goal_mode: GoalMode) -> bool:
    if goal_mode == "joint":
        return is_joint_contact(ev)
    return True


def _nearest_contact_goal(
    events_by_frame: dict[int, list[ContactEvent]],
    global_frame: int,
    hand: str,
    object_uid: int,
    *,
    search_radius: int = NEAREST_EVENT_RADIUS,
    goal_mode: GoalMode = "raw",
) -> np.ndarray | None:
    """Closest contact-surface point for one hand/object near ``global_frame``."""
    frame_events = events_by_frame.get(int(global_frame), [])
    matches = [
        ev
        for ev in frame_events
        if ev.hand == hand
        and int(ev.object_uid) == int(object_uid)
        and _event_matches_mode(ev, goal_mode)
    ]
    if matches:
        return _goal_from_event(min(matches, key=lambda e: e.distance_m))

    best: ContactEvent | None = None
    best_dist = search_radius + 1
    for delta in range(1, search_radius + 1):
        for f in (global_frame - delta, global_frame + delta):
            for ev in events_by_frame.get(int(f), []):
                if ev.hand != hand or int(ev.object_uid) != int(object_uid):
                    continue
                if not _event_matches_mode(ev, goal_mode):
                    continue
                if delta < best_dist:
                    best = ev
                    best_dist = delta
        if best is not None:
            break
    return _goal_from_event(best) if best is not None else None


def _object_bbox_for_uid(
    object_uid: int,
    objects_by_uid: dict[int, ObjectBBox3D] | None,
) -> ObjectBBox3D | None:
    if objects_by_uid is None:
        return None
    return objects_by_uid.get(int(object_uid))


APPROACH_FOV_MARGIN_RAD = 0.12
MIN_APPROACH_VISIBLE_CORNERS = 5


def _object_visible_for_approach_goal(
    obj: ObjectBBox3D,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    *,
    fov_margin_rad: float = APPROACH_FOV_MARGIN_RAD,
) -> bool:
    """Stricter than ``object_in_head_fov``: target must be squarely in view, not edge-grazing."""
    from fom.preprocessing.nymeria.goals.fov import object_visible_strict

    return object_visible_strict(
        obj,
        camera_position,
        camera_rotation,
        rgb_calib,
        fov_margin_rad=fov_margin_rad,
    )


def _approach_goal_world(
    obj: ObjectBBox3D,
    camera_position: np.ndarray,
) -> np.ndarray:
    """Camera-facing point on the object OBB (stable during navigation)."""
    return closest_point_on_obb(
        np.asarray(camera_position, dtype=np.float64),
        obj.center_world,
        obj.dimensions,
        obj.wxyz,
    )


def project_contact_goal(
    goal_world: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    projector: AriaFisheyeProjector,
    rgb_calib: RgbCameraCalibration,
    image_width: int,
    image_height: int,
    *,
    fov_margin_rad: float = 0.05,
) -> tuple[float, float] | None:
    """Project object contact point to rotated bundle RGB pixels; None if off-screen/FOV."""
    proj = projector.project_world(
        goal_world,
        camera_position,
        camera_rotation,
        fov_y_rad=rgb_calib.fov_y_rad,
        aspect=rgb_calib.aspect,
        margin_rad=fov_margin_rad,
    )
    if not proj.visible:
        return None

    su, sv = projector.scale_uv_to_image(proj.u, proj.v, image_width, image_height)
    su, sv = rotate_image_uv_cw90(su, sv, image_width, image_height)
    if not (0.0 <= su < image_width and 0.0 <= sv < image_height):
        return None
    return float(su), float(sv)


def build_fallback_goals(
    clip: CategoryClip,
    events_by_frame: dict[int, list[ContactEvent]],
    *,
    goal_mode: GoalMode = "raw",
    intervals: list[ContactInterval] | None = None,
) -> dict[tuple[str, int], np.ndarray]:
    """First contact-surface point per (hand, object_uid) at contact onset."""
    uids = _clip_object_uids(clip)
    out: dict[tuple[str, int], np.ndarray] = {}

    search_frames: list[int] = []
    if clip.chain_object_uids and clip.chain_contact_starts:
        search_frames = [int(f) for f in clip.chain_contact_starts]
    elif clip.chain_object_uids:
        search_frames = list(_infer_chain_contact_starts(clip, intervals, tuple(clip.chain_object_uids)))
    else:
        search_frames = list(range(clip.contact_start_frame_idx, clip.contact_end_frame_idx + 1))

    for f in search_frames:
        for ev in events_by_frame.get(int(f), []):
            if uids and int(ev.object_uid) not in uids:
                continue
            if not _event_matches_mode(ev, goal_mode):
                continue
            key = (ev.hand, int(ev.object_uid))
            if key not in out:
                out[key] = _goal_from_event(ev)

    if out:
        return out

    for f in range(clip.contact_start_frame_idx, clip.contact_end_frame_idx + 1):
        for ev in events_by_frame.get(f, []):
            if uids and int(ev.object_uid) not in uids:
                continue
            if not _event_matches_mode(ev, goal_mode):
                continue
            key = (ev.hand, int(ev.object_uid))
            if key not in out:
                out[key] = _goal_from_event(ev)
    return out


def active_hands_from_events(
    events_by_frame: dict[int, list[ContactEvent]],
    global_frame: int,
    object_uids: set[int],
    *,
    goal_mode: GoalMode,
) -> set[str]:
    """Per-frame active hands derived from contact events (joint mode)."""
    active: set[str] = set()
    for hand in ("left", "right"):
        frame_events = events_by_frame.get(int(global_frame), [])
        if any(
            ev.hand == hand
            and (not object_uids or int(ev.object_uid) in object_uids)
            and _event_matches_mode(ev, goal_mode)
            for ev in frame_events
        ):
            active.add(hand)
    return active


def _active_hands_from_label(label: str) -> set[str]:
    if label == "both":
        return {"left", "right"}
    if label in ("left", "right"):
        return {label}
    return set()


def _hands_for_frame(
    clip: CategoryClip,
    global_frame: int,
    in_contact: bool,
    fallback_goals: dict[tuple[str, int], np.ndarray],
    events_by_frame: dict[int, list[ContactEvent]],
    active_hands: set[str],
) -> list[tuple[str, int]]:
    """Hand/object pairs that should have a goal marker on this frame."""
    uids = _clip_object_uids(clip)
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    if in_contact:
        if not active_hands:
            return []
        for hand in sorted(active_hands):
            for uid in (uids or {clip.object_uid}):
                key = (hand, int(uid))
                if key in fallback_goals or any(
                    ev.hand == hand and int(ev.object_uid) == int(uid)
                    for ev in events_by_frame.get(int(global_frame), [])
                ):
                    if key not in seen:
                        pairs.append(key)
                        seen.add(key)
        if not pairs:
            for hand in sorted(active_hands):
                for key in fallback_goals:
                    if key[0] == hand and key not in seen:
                        pairs.append(key)
                        seen.add(key)
    else:
        hands = active_hands or _active_hands_from_label(clip.contact_hand)
        for hand in sorted(hands):
            for key, _ in fallback_goals.items():
                if key[0] == hand and key not in seen:
                    pairs.append(key)
                    seen.add(key)

    return pairs


def separate_marker_uvs(
    markers: list[tuple[str, float, float]],
    *,
    min_sep_px: float,
    image_width: int,
    image_height: int,
) -> list[tuple[str, float, float]]:
    """Push apart overlapping goal markers so L/R stay visually distinct."""
    if len(markers) < 2:
        return markers
    out = [(h, float(u), float(v)) for h, u, v in markers]
    min_sep = max(float(min_sep_px), 1.0)
    for _ in range(8):
        moved = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                _, u0, v0 = out[i]
                _, u1, v1 = out[j]
                dx, dy = u1 - u0, v1 - v0
                dist = float(np.hypot(dx, dy))
                if dist >= min_sep or dist < 1e-3:
                    continue
                push = 0.5 * (min_sep - dist)
                nx, ny = dx / dist, dy / dist
                out[i] = (out[i][0], u0 - nx * push, v0 - ny * push)
                out[j] = (out[j][0], u1 + nx * push, v1 + ny * push)
                moved = True
        if not moved:
            break
    clamped: list[tuple[str, float, float]] = []
    for hand, u, v in out:
        u = float(np.clip(u, 0.0, image_width - 1.0))
        v = float(np.clip(v, 0.0, image_height - 1.0))
        clamped.append((hand, u, v))
    return clamped


def build_goal_points_for_frame(
    clip: CategoryClip,
    global_frame: int,
    events_by_frame: dict[int, list[ContactEvent]],
    fallback_goals: dict[tuple[str, int], np.ndarray],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    projector: AriaFisheyeProjector,
    rgb_calib: RgbCameraCalibration,
    image_width: int,
    image_height: int,
    *,
    in_contact: bool,
    active_hands: set[str],
    fov_margin_rad: float = 0.05,
    goal_mode: GoalMode = "raw",
    goal_pairs: list[tuple[str, int]] | None = None,
    objects_by_uid: dict[int, ObjectBBox3D] | None = None,
    require_object_in_fov: bool = True,
) -> list[tuple[str, float, float]]:
    """Return [(hand, u, v), ...] for per-hand contact goals visible in head FOV."""
    if clip.category == "cat2_navigation":
        return []

    raw: list[tuple[str, float, float]] = []
    seen_hands: set[str] = set()

    pairs = goal_pairs
    if pairs is None:
        pairs = _hands_for_frame(
            clip, global_frame, in_contact, fallback_goals, events_by_frame, active_hands
        )

    for hand, object_uid in pairs:
        if hand in seen_hands:
            continue

        goal = _nearest_contact_goal(
            events_by_frame,
            global_frame,
            hand,
            object_uid,
            goal_mode=goal_mode,
        )
        if goal is None:
            goal = fallback_goals.get((hand, object_uid))
        if goal is None:
            continue

        obj = _object_bbox_for_uid(object_uid, objects_by_uid)
        if require_object_in_fov and obj is not None:
            strict_fov = not in_contact
            if strict_fov and not _object_visible_for_approach_goal(
                obj,
                camera_position,
                camera_rotation,
                rgb_calib,
                fov_margin_rad=max(float(fov_margin_rad), APPROACH_FOV_MARGIN_RAD),
            ):
                continue
            if not in_contact:
                goal = _approach_goal_world(obj, camera_position)

        uv = project_contact_goal(
            goal,
            camera_position,
            camera_rotation,
            projector,
            rgb_calib,
            image_width,
            image_height,
            fov_margin_rad=fov_margin_rad,
        )
        if uv is not None:
            raw.append((hand, uv[0], uv[1]))
            seen_hands.add(hand)

    if len(raw) >= 2:
        sep = max(24.0, 2.2 * GOAL_RADIUS)
        raw = separate_marker_uvs(
            raw,
            min_sep_px=sep,
            image_width=image_width,
            image_height=image_height,
        )
    return raw


def _draw_goal_dot(draw: ImageDraw.ImageDraw, u: float, v: float, hand: str, radius: int) -> None:
    color = HAND_COLORS.get(hand, HAND_COLORS["unknown"])
    ui, vi = int(round(u)), int(round(v))
    draw.ellipse(
        [ui - radius, vi - radius, ui + radius, vi + radius],
        fill=color,
        outline=(255, 255, 255),
        width=2,
    )


def _draw_hand_border(img: Image.Image, hands: set[str]) -> Image.Image:
    if not hands or hands == {"unknown"}:
        return img
    if "left" in hands and "right" in hands:
        colors = [HAND_COLORS["left"], HAND_COLORS["right"]]
    elif "left" in hands:
        colors = [HAND_COLORS["left"]]
    elif "right" in hands:
        colors = [HAND_COLORS["right"]]
    else:
        return img

    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    bw = BORDER_WIDTH
    if len(colors) == 2:
        half = w // 2
        draw.rectangle([0, 0, half, bw - 1], fill=colors[0])
        draw.rectangle([w - bw, 0, w - 1, h - 1], outline=colors[1], width=bw)
    else:
        for i in range(bw):
            draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=colors[0])
    return out


def annotate_category_frame(
    frame: np.ndarray,
    *,
    frame_idx: int,
    num_frames: int,
    clip: CategoryClip,
    goal_points: list[tuple[str, float, float]],
    contact_hands: set[str],
    in_contact: bool,
    uv_scale: float,
    goal_radius: int,
    goal_mode: GoalMode = "raw",
    chain_target_uid: int | None = None,
) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)

    scaled_goals: list[tuple[str, float, float]] = [
        (hand, u * uv_scale, v * uv_scale) for hand, u, v in goal_points
    ]
    r = max(4, int(round(goal_radius * uv_scale)))
    for hand, u, v in scaled_goals:
        _draw_goal_dot(draw, u, v, hand, r)

    if in_contact and contact_hands:
        img = _draw_hand_border(img, contact_hands)

    category_label = {
        "cat1_fov_navigate": "Cat1: FOV navigate",
        "cat2_navigation": "Cat2: Navigation",
        "cat3_sustained_interaction": "Cat3: Interaction",
        "cat4_multi_object_chain": "Cat4: Multi-object chain",
    }.get(clip.category, clip.category)
    mode_label = "joint hand goals" if goal_mode == "joint" else "raw contact goals"

    lines = [
        category_label,
        mode_label,
        f"frame {frame_idx + 1}/{num_frames}",
        f"{clip.instance_name or clip.category_name} (obj {clip.object_uid})",
    ]
    if scaled_goals:
        hands_shown = sorted({h for h, _, _ in scaled_goals})
        lines.append("goals: " + ", ".join(hands_shown))
    elif chain_target_uid is not None:
        lines.append("goal hidden (obj not in FOV)")
    if in_contact and contact_hands:
        lines.append("contact: " + ", ".join(sorted(contact_hands)))
    if clip.chain_object_uids:
        lines.append(f"chain: {list(clip.chain_object_uids)}")
    if chain_target_uid is not None:
        lines.append(f"target obj: #{chain_target_uid}")

    y = 8
    for line in lines:
        draw.text((8, y), line, fill=(0, 0, 0))
        draw.text((9, y + 1), line, fill=(255, 255, 255))
        y += 16

    h = img.size[1]
    legend_y = h - 22
    if "left" in {h for h, _, _ in scaled_goals} or "left" in contact_hands:
        draw.ellipse([8, legend_y, 22, legend_y + 14], fill=HAND_COLORS["left"])
        draw.text((28, legend_y), "L", fill=HAND_COLORS["left"])
    if "right" in {h for h, _, _ in scaled_goals} or "right" in contact_hands:
        draw.ellipse([48, legend_y, 62, legend_y + 14], fill=HAND_COLORS["right"])
        draw.text((68, legend_y), "R", fill=HAND_COLORS["right"])

    return img


def save_category_gif(
    rgb: np.ndarray,
    out_path: Path,
    clip: CategoryClip,
    *,
    fps: float,
    frame_hands: list[str],
    global_frames: list[int],
    camera_indices: list[int],
    events_by_frame: dict[int, list[ContactEvent]],
    fallback_goals: dict[tuple[str, int], np.ndarray],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    projector: AriaFisheyeProjector,
    rgb_calib: RgbCameraCalibration,
    native_image_width: int,
    native_image_height: int,
    uv_scale: float = 1.0,
    goal_radius: int = GOAL_RADIUS,
    fov_margin_rad: float = 0.05,
    goal_mode: GoalMode = "raw",
    intervals: list[ContactInterval] | None = None,
    objects_by_uid: dict[int, ObjectBBox3D] | None = None,
    require_object_in_fov: bool = True,
) -> None:
    duration_ms = max(int(round(1000.0 / max(fps, 0.1))), 1)
    num_frames = int(rgb.shape[0])
    pil_frames: list[Image.Image] = []

    for i in range(num_frames):
        global_f = int(global_frames[i]) if i < len(global_frames) else clip.start_frame_idx + i
        cam_i = int(camera_indices[i]) if i < len(camera_indices) else global_f - clip.start_frame_idx
        cam_i = min(max(cam_i, 0), len(camera_position) - 1)
        in_contact = clip.contact_start_frame_idx <= global_f <= clip.contact_end_frame_idx

        contact_hands: set[str] = set()
        active_hands: set[str] = set()
        goal_pairs: list[tuple[str, int]] | None = None
        chain_target_uid: int | None = None

        chain_state = resolve_frame_goal_state(
            clip,
            global_f,
            intervals=intervals,
            events_by_frame=events_by_frame,
            goal_mode=goal_mode,
        )
        if chain_state is not None:
            in_contact = chain_state.in_contact
            contact_hands = set(chain_state.contact_hands)
            goal_pairs = list(chain_state.goal_pairs)
            chain_target_uid = chain_state.chain_target_uid
        elif goal_mode == "joint" and in_contact:
            clip_uids = _clip_object_uids(clip)
            active_hands = active_hands_from_events(
                events_by_frame, global_f, clip_uids, goal_mode="joint"
            )
            contact_hands = set(active_hands)
        elif i < len(frame_hands) and frame_hands[i]:
            h = frame_hands[i]
            if h == "both":
                contact_hands = {"left", "right"}
                active_hands = {"left", "right"}
            elif h in ("left", "right"):
                contact_hands = {h}
                active_hands = {h}
        if not in_contact and chain_state is None:
            active_hands = _active_hands_from_label(clip.contact_hand)

        goal_points = build_goal_points_for_frame(
            clip,
            global_f,
            events_by_frame,
            fallback_goals,
            camera_position[cam_i],
            camera_rotation[cam_i],
            projector,
            rgb_calib,
            native_image_width,
            native_image_height,
            in_contact=in_contact,
            active_hands=active_hands,
            fov_margin_rad=fov_margin_rad,
            goal_mode=goal_mode,
            goal_pairs=goal_pairs,
            objects_by_uid=objects_by_uid,
            require_object_in_fov=require_object_in_fov,
        )

        pil_frames.append(
            annotate_category_frame(
                rgb[i],
                frame_idx=i,
                num_frames=num_frames,
                clip=clip,
                goal_points=goal_points,
                contact_hands=contact_hands if in_contact else set(),
                in_contact=in_contact,
                uv_scale=uv_scale,
                goal_radius=goal_radius,
                goal_mode=goal_mode,
                chain_target_uid=chain_target_uid,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
