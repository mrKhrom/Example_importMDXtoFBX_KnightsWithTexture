"""Pivots, local TRS, quaternions, and axis conversion."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Mat4 = List[List[float]]

IDENTITY_QUAT: Quat = (0.0, 0.0, 0.0, 1.0)
ONES: Vec3 = (1.0, 1.0, 1.0)
ZERO: Vec3 = (0.0, 0.0, 0.0)


def vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vmul(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def vscale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vlen(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def vlerp(a: Sequence[float], b: Sequence[float], t: float) -> Tuple[float, ...]:
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def qdot(a: Quat, b: Quat) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def qneg(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], -q[3])


def qnormalize(q: Quat) -> Quat:
    n = vlen(q)
    if n <= 1e-12:
        return IDENTITY_QUAT
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def qmul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qslerp(a: Quat, b: Quat, t: float) -> Quat:
    a = qnormalize(a)
    b = qnormalize(b)
    d = qdot(a, b)
    if d < 0.0:
        b = qneg(b)
        d = -d
    if d > 0.9995:
        return qnormalize(
            (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
                a[3] + (b[3] - a[3]) * t,
            )
        )
    d = max(-1.0, min(1.0, d))
    theta = math.acos(d)
    s = math.sin(theta)
    w1 = math.sin((1.0 - t) * theta) / s
    w2 = math.sin(t * theta) / s
    return (
        a[0] * w1 + b[0] * w2,
        a[1] * w1 + b[1] * w2,
        a[2] * w1 + b[2] * w2,
        a[3] * w1 + b[3] * w2,
    )


def q_to_mat3(q: Quat) -> List[List[float]]:
    x, y, z, w = qnormalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def mat3_to_q(m: Sequence[Sequence[float]]) -> Quat:
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return qnormalize((x, y, z, w))


def identity4() -> Mat4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = (
                a[i][0] * b[0][j]
                + a[i][1] * b[1][j]
                + a[i][2] * b[2][j]
                + a[i][3] * b[3][j]
            )
    return out


def mat4_mul_vec3(m: Mat4, v: Vec3, w: float = 1.0) -> Vec3:
    x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * w
    y = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * w
    z = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * w
    if w == 0.0:
        return (x, y, z)
    ww = m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * w
    if abs(ww) > 1e-12 and abs(ww - 1.0) > 1e-12:
        return (x / ww, y / ww, z / ww)
    return (x, y, z)


def mat4_from_trs(t: Vec3, r: Quat, s: Vec3) -> Mat4:
    rot = q_to_mat3(r)
    m = identity4()
    for i in range(3):
        m[i][0] = rot[i][0] * s[0]
        m[i][1] = rot[i][1] * s[1]
        m[i][2] = rot[i][2] * s[2]
        m[i][3] = t[i]
    return m


def mat4_invert(m: Mat4) -> Mat4:
    """Invert an affine 4x4 (rotation/scale + translation)."""
    r = [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ]
    det = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(det) < 1e-12:
        return identity4()
    invdet = 1.0 / det
    invr = [
        [
            (r[1][1] * r[2][2] - r[1][2] * r[2][1]) * invdet,
            (r[0][2] * r[2][1] - r[0][1] * r[2][2]) * invdet,
            (r[0][1] * r[1][2] - r[0][2] * r[1][1]) * invdet,
        ],
        [
            (r[1][2] * r[2][0] - r[1][0] * r[2][2]) * invdet,
            (r[0][0] * r[2][2] - r[0][2] * r[2][0]) * invdet,
            (r[0][2] * r[1][0] - r[0][0] * r[1][2]) * invdet,
        ],
        [
            (r[1][0] * r[2][1] - r[1][1] * r[2][0]) * invdet,
            (r[0][1] * r[2][0] - r[0][0] * r[2][1]) * invdet,
            (r[0][0] * r[1][1] - r[0][1] * r[1][0]) * invdet,
        ],
    ]
    t = (m[0][3], m[1][3], m[2][3])
    it = (
        -(invr[0][0] * t[0] + invr[0][1] * t[1] + invr[0][2] * t[2]),
        -(invr[1][0] * t[0] + invr[1][1] * t[1] + invr[1][2] * t[2]),
        -(invr[2][0] * t[0] + invr[2][1] * t[1] + invr[2][2] * t[2]),
    )
    out = identity4()
    for i in range(3):
        out[i][0], out[i][1], out[i][2] = invr[i]
        out[i][3] = it[i]
    return out


def mat4_decompose(m: Mat4) -> Tuple[Vec3, Quat, Vec3]:
    t = (m[0][3], m[1][3], m[2][3])
    sx = vlen((m[0][0], m[1][0], m[2][0]))
    sy = vlen((m[0][1], m[1][1], m[2][1]))
    sz = vlen((m[0][2], m[1][2], m[2][2]))
    # Reflection: if det < 0, flip one scale axis.
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if det < 0.0:
        sx = -sx
    rot = [
        [m[0][0] / sx if abs(sx) > 1e-12 else 0.0,
         m[0][1] / sy if abs(sy) > 1e-12 else 0.0,
         m[0][2] / sz if abs(sz) > 1e-12 else 0.0],
        [m[1][0] / sx if abs(sx) > 1e-12 else 0.0,
         m[1][1] / sy if abs(sy) > 1e-12 else 0.0,
         m[1][2] / sz if abs(sz) > 1e-12 else 0.0],
        [m[2][0] / sx if abs(sx) > 1e-12 else 0.0,
         m[2][1] / sy if abs(sy) > 1e-12 else 0.0,
         m[2][2] / sz if abs(sz) > 1e-12 else 0.0],
    ]
    return t, mat3_to_q(rot), (sx, sy, sz)


def mat4_to_col_major(m: Mat4) -> List[float]:
    """FBX ASCII stores 4x4 as column-major (translation at 12,13,14)."""
    out: List[float] = []
    for c in range(4):
        for r in range(4):
            out.append(m[r][c])
    return out


def hermite(t: float, p0: float, p1: float, out_tan: float, in_tan: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * p0
        + (t3 - 2.0 * t2 + t) * out_tan
        + (-2.0 * t3 + 3.0 * t2) * p1
        + (t3 - t2) * in_tan
    )


def bezier(t: float, p0: float, p1: float, out_tan: float, in_tan: float) -> float:
    it = 1.0 - t
    return (
        it * it * it * p0
        + 3.0 * it * it * t * out_tan
        + 3.0 * it * t * t * in_tan
        + t * t * t * p1
    )


def interpolate_component(
    kind: int,
    t: float,
    a: float,
    b: float,
    a_out: Optional[float],
    b_in: Optional[float],
) -> float:
    if kind <= 1:
        return a + (b - a) * t
    ao = 0.0 if a_out is None else a_out
    bi = 0.0 if b_in is None else b_in
    if kind == 2:
        return hermite(t, a, b, ao, bi)
    return bezier(t, a, b, ao, bi)


def interpolate_value(
    kind: int,
    t: float,
    a,
    b,
    a_out=None,
    b_in=None,
    is_quat: bool = False,
):
    if is_quat:
        if kind == 0:
            return a
        # Component hermite/bezier is what WC3 does for quats; for Linear use slerp.
        if kind == 1 or a_out is None:
            return qslerp(a, b, t)
        out = []
        for i in range(4):
            ao = a_out[i] if a_out is not None else None
            bi = b_in[i] if b_in is not None else None
            out.append(interpolate_component(kind, t, a[i], b[i], ao, bi))
        return qnormalize(tuple(out))  # type: ignore[arg-type]
    if isinstance(a, tuple):
        res = []
        for i in range(len(a)):
            ao = a_out[i] if a_out is not None else None
            bi = b_in[i] if b_in is not None else None
            res.append(interpolate_component(kind, t, a[i], b[i], ao, bi))
        return tuple(res)
    return interpolate_component(
        kind,
        t,
        float(a),
        float(b),
        None if a_out is None else float(a_out),
        None if b_in is None else float(b_in),
    )


def rest_local_translation(
    pivot: Vec3, parent_pivot: Optional[Vec3]
) -> Vec3:
    if parent_pivot is None:
        return pivot
    return vsub(pivot, parent_pivot)


def axis_matrix(mode: str) -> Mat4:
    """Linear map applied to points.

    raw:    identity (WC3 Z-up, X forward, Y right)
    unity:  (x, y, z) -> (y, z, x)  so Unity Y-up, character faces +Z.
            This is a rotation (det +1), so Unity will not double-mirror.
    """
    m = identity4()
    if mode == "raw":
        return m
    # (x, y, z) -> (y, z, x)
    m[0] = [0.0, 1.0, 0.0, 0.0]
    m[1] = [0.0, 0.0, 1.0, 0.0]
    m[2] = [1.0, 0.0, 0.0, 0.0]
    return m


def apply_axis_point(p: Vec3, mode: str) -> Vec3:
    if mode == "raw":
        return p
    # (x, y, z) -> (y, z, x)
    return (p[1], p[2], p[0])


def apply_axis_matrix(local: Mat4, mode: str) -> Mat4:
    if mode == "raw":
        return local
    m = axis_matrix(mode)
    # L' = M L M^{-1}
    return mat4_mul(m, mat4_mul(local, mat4_invert(m)))


def quat_to_euler_xyz_deg(q: Quat) -> Vec3:
    x, y, z, w = qnormalize(q)
    # roll (x)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    # pitch (y)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (z)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def continue_euler(prev: Optional[Vec3], curr: Vec3) -> Vec3:
    if prev is None:
        return curr
    out = []
    for p, c in zip(prev, curr):
        # unwrap to nearest equivalent angle
        delta = c - p
        delta = (delta + 180.0) % 360.0 - 180.0
        out.append(p + delta)
    return (out[0], out[1], out[2])


def flip_winding(indices: Iterable[int]) -> List[int]:
    idx = list(indices)
    out: List[int] = []
    for i in range(0, len(idx), 3):
        if i + 2 >= len(idx):
            break
        out.extend((idx[i], idx[i + 2], idx[i + 1]))
    return out
