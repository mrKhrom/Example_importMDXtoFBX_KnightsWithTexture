"""Track evaluation, sequence slicing, and clip baking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from mdx_parser import INTERP_NONE, Keyframe, Track
from xform import (
    IDENTITY_QUAT,
    Quat,
    Vec3,
    continue_euler,
    interpolate_value,
    qdot,
    qneg,
    qnormalize,
    quat_to_euler_xyz_deg,
)

NO_GLOBAL = None


@dataclass
class Sample:
    time_s: float
    translation: Vec3
    rotation: Quat
    scaling: Vec3
    euler_deg: Vec3


@dataclass
class BakedNode:
    node_id: int
    samples: List[Sample] = field(default_factory=list)


@dataclass
class BakedVisibility:
    mesh_index: int
    times_s: List[float]
    values: List[float]


@dataclass
class BakedClip:
    name: str
    duration_s: float
    looping: bool
    nodes: List[BakedNode]
    visibility: List[BakedVisibility] = field(default_factory=list)


def find_track(tracks: Sequence[Track], tag: str, allow_global: bool = False) -> Optional[Track]:
    for tr in tracks:
        if tr.tag != tag:
            continue
        if tr.global_sequence is not None and not allow_global:
            continue
        if tr.global_sequence is None and allow_global:
            continue
        return tr
    return None


def _key_index(keys: Sequence[Keyframe], time: int) -> Tuple[int, int, float]:
    """Return (i0, i1, local_t) for sampling at `time`."""
    if not keys:
        return 0, 0, 0.0
    if time <= keys[0].time:
        return 0, 0, 0.0
    if time >= keys[-1].time:
        last = len(keys) - 1
        return last, last, 0.0
    lo = 0
    hi = len(keys) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if keys[mid].time <= time:
            lo = mid
        else:
            hi = mid
    span = keys[hi].time - keys[lo].time
    t = 0.0 if span <= 0 else (time - keys[lo].time) / float(span)
    return lo, hi, t


def eval_track(
    track: Optional[Track],
    time: int,
    default,
    is_quat: bool = False,
    clip_start: Optional[int] = None,
    clip_end: Optional[int] = None,
):
    """Evaluate a track.

    When clip_start/clip_end are set, keys outside that Warcraft sequence are
    ignored. Missing in-range keys fall back to `default` (identity offset).
    This matches isolated Unity clips and prevents Death/Decay poses from
    leaking into Stand/Attack/Portrait/Spell.
    """
    if track is None or not track.keys:
        return default
    keys = track.keys
    if clip_start is not None and clip_end is not None:
        keys = [k for k in keys if clip_start <= k.time <= clip_end]
        if not keys:
            return default
    i0, i1, t = _key_index(keys, time)
    k0 = keys[i0]
    k1 = keys[i1]
    if i0 == i1 or track.interpolation == INTERP_NONE:
        return k0.value
    return interpolate_value(
        track.interpolation,
        t,
        k0.value,
        k1.value,
        k0.out_tan,
        k1.in_tan,
        is_quat=is_quat,
    )


def eval_translation(
    tracks: Sequence[Track],
    time: int,
    clip_start: Optional[int] = None,
    clip_end: Optional[int] = None,
) -> Vec3:
    tr = find_track(tracks, "KGTR")
    val = eval_track(tr, time, (0.0, 0.0, 0.0), clip_start=clip_start, clip_end=clip_end)
    return (float(val[0]), float(val[1]), float(val[2]))  # type: ignore[index]


def eval_rotation(
    tracks: Sequence[Track],
    time: int,
    clip_start: Optional[int] = None,
    clip_end: Optional[int] = None,
) -> Quat:
    tr = find_track(tracks, "KGRT")
    val = eval_track(
        tr, time, IDENTITY_QUAT, is_quat=True, clip_start=clip_start, clip_end=clip_end
    )
    return qnormalize((float(val[0]), float(val[1]), float(val[2]), float(val[3])))  # type: ignore[index]


def eval_scaling(
    tracks: Sequence[Track],
    time: int,
    clip_start: Optional[int] = None,
    clip_end: Optional[int] = None,
) -> Vec3:
    tr = find_track(tracks, "KGSC")
    val = eval_track(tr, time, (1.0, 1.0, 1.0), clip_start=clip_start, clip_end=clip_end)
    return (float(val[0]), float(val[1]), float(val[2]))  # type: ignore[index]


def eval_alpha(
    tracks: Sequence[Track],
    time: int,
    default: float = 1.0,
    clip_start: Optional[int] = None,
    clip_end: Optional[int] = None,
) -> float:
    tr = find_track(tracks, "KGAO")
    if tr is None:
        tr = find_track(tracks, "KMTA")
    val = eval_track(tr, time, default, clip_start=clip_start, clip_end=clip_end)
    if isinstance(val, tuple):
        return float(val[0])
    return float(val)


def sanitize_clip_name(name: str, used: Optional[set] = None) -> str:
    out = []
    prev_us = False
    for ch in name.strip():
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    cleaned = "".join(out).strip("_")
    if not cleaned:
        cleaned = "Clip"
    if used is None:
        return cleaned
    base = cleaned
    n = 2
    while cleaned in used:
        cleaned = "%s_%d" % (base, n)
        n += 1
    used.add(cleaned)
    return cleaned


def frame_times_ms(start: int, end: int, fps: float, bake: bool) -> List[int]:
    if end < start:
        end = start
    times = {start, end}
    if bake and fps > 0:
        duration = (end - start) / 1000.0
        frames = int(math.floor(duration * fps))
        step = 1000.0 / fps
        for k in range(0, frames + 1):
            t = int(round(start + k * step))
            if t > end:
                t = end
            times.add(t)
    return sorted(times)


def collect_key_times(tracks: Sequence[Track], start: int, end: int) -> List[int]:
    times = {start, end}
    for tr in tracks:
        if tr.global_sequence is not None:
            continue
        for k in tr.keys:
            if start <= k.time <= end:
                times.add(k.time)
    return sorted(times)


def local_trs(
    rest_t: Vec3,
    tracks: Sequence[Track],
    time: int,
    clip_start: Optional[int] = None,
    clip_end: Optional[int] = None,
) -> Tuple[Vec3, Quat, Vec3]:
    off_t = eval_translation(tracks, time, clip_start, clip_end)
    rot = eval_rotation(tracks, time, clip_start, clip_end)
    scl = eval_scaling(tracks, time, clip_start, clip_end)
    t = (rest_t[0] + off_t[0], rest_t[1] + off_t[1], rest_t[2] + off_t[2])
    return t, rot, scl


def _is_almost(a: Sequence[float], b: Sequence[float], eps: float = 1e-6) -> bool:
    return all(abs(x - y) <= eps for x, y in zip(a, b))


def bake_node_clip(
    node_id: int,
    rest_t: Vec3,
    tracks: Sequence[Track],
    start: int,
    end: int,
    times_ms: Sequence[int],
    force: bool = True,
) -> Optional[BakedNode]:
    """Bake a node.

    `force=True` (default) always writes curves so Unity resets the bone when
    switching takes. Otherwise Death/Decay offsets linger into Stand/Attack.
    """
    samples: List[Sample] = []
    prev_q: Optional[Quat] = None
    prev_e: Optional[Vec3] = None
    changed = False
    bind_t, bind_r, bind_s = rest_t, IDENTITY_QUAT, (1.0, 1.0, 1.0)

    for t_ms in times_ms:
        t, r, s = local_trs(rest_t, tracks, t_ms, clip_start=start, clip_end=end)
        if prev_q is not None and qdot(prev_q, r) < 0.0:
            r = qneg(r)
        prev_q = r
        e = continue_euler(prev_e, quat_to_euler_xyz_deg(r))
        prev_e = e
        samples.append(
            Sample((t_ms - start) / 1000.0, t, r, s, e)
        )
        if (
            not _is_almost(t, bind_t)
            or not _is_almost(r, bind_r)
            or not _is_almost(s, bind_s)
        ):
            changed = True

    if not changed and not force:
        return None

    # Collapse constant-after-first curves? Keep samples; writer may compress.
    # If only 2 unique values and they're equal, keep first+last.
    if len(samples) > 2:
        same = True
        first = samples[0]
        for s in samples[1:]:
            if (
                not _is_almost(s.translation, first.translation)
                or not _is_almost(s.rotation, first.rotation)
                or not _is_almost(s.scaling, first.scaling)
            ):
                same = False
                break
        if same:
            samples = [samples[0], samples[-1]]

    return BakedNode(node_id, samples)


def bake_visibility(
    mesh_index: int,
    tracks: Sequence[Track],
    default_alpha: float,
    start: int,
    end: int,
) -> Optional[BakedVisibility]:
    tr = find_track(tracks, "KGAO")
    times = {start, end}
    if tr is not None:
        for k in tr.keys:
            if start <= k.time <= end:
                times.add(k.time)
            # Step tracks: also include the instant before a change if needed.
    times_ms = sorted(times)
    vals = []
    ts = []
    for t in times_ms:
        a = eval_alpha(tracks, t, default_alpha, clip_start=start, clip_end=end)
        vis = 1.0 if a >= 0.5 else 0.0
        ts.append((t - start) / 1000.0)
        vals.append(vis)
    if not vals:
        return None
    # If constantly visible (1), skip — default visibility is on.
    if all(v >= 0.5 for v in vals) and default_alpha >= 0.5:
        return None
    return BakedVisibility(mesh_index, ts, vals)
