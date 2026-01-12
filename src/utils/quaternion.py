from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def _stack_xyzw(qw, qx, qy, qz) -> tuple[np.ndarray, bool]:
    qw = np.asarray(qw, dtype=np.float64)
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)
    qz = np.asarray(qz, dtype=np.float64)
    qw, qx, qy, qz = np.broadcast_arrays(qw, qx, qy, qz)
    if qw.shape == ():
        return np.array([qx.item(), qy.item(), qz.item(), qw.item()], dtype=np.float64), True
    return np.stack([qx, qy, qz, qw], axis=-1), False


def _mask_quats(q_xyzw: np.ndarray):
    if q_xyzw.ndim == 1:
        finite = bool(np.all(np.isfinite(q_xyzw)))
        norm = float(np.linalg.norm(q_xyzw))
        return finite and norm > 0.0, finite and norm == 0.0, not finite
    flat = q_xyzw.reshape(-1, 4)
    finite = np.all(np.isfinite(flat), axis=1)
    norms = np.linalg.norm(flat, axis=1)
    valid = finite & (norms > 0.0)
    zero = finite & (norms == 0.0)
    invalid = ~finite
    return valid, zero, invalid


def _replace_zero_quats(q_xyzw: np.ndarray) -> np.ndarray:
    # SciPy rejects zero-norm quaternions; map them to identity rotations.
    if q_xyzw.ndim == 1:
        if np.all(q_xyzw == 0.0):
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q_xyzw
    norms = np.linalg.norm(q_xyzw.reshape(-1, 4), axis=1)
    mask = norms == 0.0
    if not np.any(mask):
        return q_xyzw
    out = q_xyzw.reshape(-1, 4).copy()
    out[mask] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return out.reshape(q_xyzw.shape)


def rotation_from_wxyz(qw, qx, qy, qz) -> R:
    q_xyzw, _ = _stack_xyzw(qw, qx, qy, qz)
    q_xyzw = _replace_zero_quats(q_xyzw)
    return R.from_quat(q_xyzw)


def normalize_quat_arrays(qw, qx, qy, qz):
    q_xyzw, scalar = _stack_xyzw(qw, qx, qy, qz)
    if scalar:
        valid, zero, invalid = _mask_quats(q_xyzw)
        if invalid:
            return float("nan"), float("nan"), float("nan"), float("nan")
        if zero:
            return 0.0, 0.0, 0.0, 0.0
        rot = R.from_quat(q_xyzw)
        q_xyzw = rot.as_quat()
        return float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])

    shape = q_xyzw.shape[:-1]
    flat = q_xyzw.reshape(-1, 4)
    valid, zero, invalid = _mask_quats(q_xyzw)
    out = np.full_like(flat, np.nan)
    if np.any(zero):
        out[zero] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if np.any(valid):
        rot = R.from_quat(flat[valid])
        out[valid] = rot.as_quat()
    out = out.reshape(shape + (4,))
    return out[..., 3], out[..., 0], out[..., 1], out[..., 2]


def rotate_vec_with_quat(qw, qx, qy, qz, vx, vy, vz) -> np.ndarray:
    q_xyzw, scalar = _stack_xyzw(qw, qx, qy, qz)
    v = np.array([vx, vy, vz], dtype=np.float64)
    if scalar:
        valid, zero, invalid = _mask_quats(q_xyzw)
        if invalid:
            return np.full((1, 3), np.nan, dtype=np.float64)
        if zero:
            return v.reshape(1, 3)
        return R.from_quat(q_xyzw).apply(v).reshape(1, 3)

    shape = q_xyzw.shape[:-1]
    flat = q_xyzw.reshape(-1, 4)
    valid, zero, invalid = _mask_quats(q_xyzw)
    out = np.full((flat.shape[0], 3), np.nan, dtype=np.float64)
    if np.any(zero):
        out[zero] = v
    if np.any(valid):
        out[valid] = R.from_quat(flat[valid]).apply(v)
    return out.reshape(shape + (3,))


def euler_zyx_from_wxyz(qw, qx, qy, qz, *, degrees: bool = False):
    q_xyzw, scalar = _stack_xyzw(qw, qx, qy, qz)
    if scalar:
        valid, zero, invalid = _mask_quats(q_xyzw)
        if invalid:
            return float("nan"), float("nan"), float("nan")
        if zero:
            return 0.0, 0.0, 0.0
        yaw, pitch, roll = R.from_quat(q_xyzw).as_euler("ZYX", degrees=degrees)
        return yaw, pitch, roll

    shape = q_xyzw.shape[:-1]
    flat = q_xyzw.reshape(-1, 4)
    valid, zero, invalid = _mask_quats(q_xyzw)
    out = np.full((flat.shape[0], 3), np.nan, dtype=np.float64)
    if np.any(zero):
        out[zero] = 0.0
    if np.any(valid):
        out[valid] = R.from_quat(flat[valid]).as_euler("ZYX", degrees=degrees)
    out = out.reshape(shape + (3,))
    return out[..., 0], out[..., 1], out[..., 2]


def yaw_from_wxyz(qw, qx, qy, qz):
    yaw, _, _ = euler_zyx_from_wxyz(qw, qx, qy, qz, degrees=False)
    return yaw
