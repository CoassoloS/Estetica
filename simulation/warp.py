"""
simulation/warp.py — Warp piecewise-affine de la foto guiado por la malla.

Cada triángulo de la malla (vértices faciales + un anillo exterior fijo)
se deforma con una transformación afín exacta: los landmarks terminan
exactamente donde indica la simulación y el resto de la imagen queda
intacto. Se usa mapeo inverso con un único `cv2.remap`, sin costuras.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import Delaunay

from simulation.mesh import FACE_OVAL, FACE_VERTEX_COUNT, FaceMesh3D


RING_SCALE = 1.45           # Anillo fijo: óvalo facial expandido desde el centro
BORDER_POINTS_PER_EDGE = 5
MIN_AREA_RATIO = 0.15       # Triángulos más comprimidos que esto se consideran plegados
MOUTH_OPEN_MIN_PX = 3.0     # Apertura mínima para tratar el interior bucal por separado

# Borde interno de los labios, de comisura a comisura
LIP_INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIP_INNER_LOWER = [95, 88, 178, 87, 14, 317, 402, 318, 324]


@dataclass
class WarpResult:
    image: np.ndarray
    applied_fraction: float     # 1.0 = desplazamiento completo; < 1 si se limitó para evitar pliegues


def warp_image(
    image: np.ndarray,
    mesh: FaceMesh3D,
    displacement: np.ndarray,
) -> WarpResult:
    """
    Deformar la imagen según el desplazamiento de la malla.

    Args:
        image: Imagen BGR donde se detectó la malla.
        mesh: Malla en reposo.
        displacement: Desplazamiento (N, 3) en px; se usa la componente xy.
    """
    h, w = image.shape[:2]
    face_disp = displacement[:FACE_VERTEX_COUNT, :2]

    if np.abs(face_disp).max(initial=0.0) < 0.05:
        return WarpResult(image.copy(), 1.0)

    src = np.vstack([mesh.xy[:FACE_VERTEX_COUNT], _fixed_ring(mesh, w, h)])
    disp = np.zeros_like(src)
    disp[:FACE_VERTEX_COUNT] = face_disp

    triangles = Delaunay(src).simplices
    fraction = _safe_fraction(src, disp, triangles)
    dst = src + disp * fraction

    moved = np.abs(disp).max(axis=1) > 0.01
    active = triangles[moved[triangles].any(axis=1)]
    if len(active) == 0:
        return WarpResult(image.copy(), fraction)

    result = image.copy()
    x0, y0, x1, y1 = _bbox(dst[active.ravel()], src[active.ravel()], w, h)
    labels = np.zeros((y1 - y0, x1 - x0), dtype=np.int32)
    inverse = np.zeros((len(active) + 1, 2, 3), dtype=np.float64)
    inverse[0] = [[1, 0, 0], [0, 1, 0]]     # Etiqueta 0: identidad

    offset = np.array([x0, y0], dtype=np.float64)
    for k, tri in enumerate(active, start=1):
        inverse[k] = cv2.getAffineTransform(
            dst[tri].astype(np.float32), src[tri].astype(np.float32)
        )
        poly = np.round((dst[tri] - offset) * 16).astype(np.int32)
        cv2.fillConvexPoly(labels, poly, k, lineType=cv2.LINE_8, shift=4)

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    m = inverse[labels]
    map_x = (m[..., 0, 0] * xx + m[..., 0, 1] * yy + m[..., 0, 2]).astype(np.float32)
    map_y = (m[..., 1, 0] * xx + m[..., 1, 1] * yy + m[..., 1, 2]).astype(np.float32)

    result[y0:y1, x0:x1] = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    _restore_mouth_interior(result, image, src, dst)
    return WarpResult(result, fraction)


def _restore_mouth_interior(
    result: np.ndarray,
    original: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> None:
    """
    Con la boca abierta, los dientes no se deforman: el labio desplazado
    se superpone sobre el interior bucal original (ej. sonrisa gingival).
    """
    opening = np.linalg.norm(src[13] - src[14])
    if opening < MOUTH_OPEN_MIN_PX:
        return

    polygon = np.vstack([dst[LIP_INNER_UPPER], dst[LIP_INNER_LOWER[::-1]]])
    mask = np.zeros(result.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon * 16).astype(np.int32)], 255,
                 lineType=cv2.LINE_AA, shift=4)
    alpha = (cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0)[..., None]
    blended = alpha * original.astype(np.float32) + (1.0 - alpha) * result.astype(np.float32)
    np.copyto(result, np.clip(blended, 0, 255).astype(np.uint8))


def _fixed_ring(mesh: FaceMesh3D, w: int, h: int) -> np.ndarray:
    """Puntos fijos: óvalo facial expandido + borde de la imagen."""
    oval = mesh.xy[FACE_OVAL]
    center = mesh.xy[:FACE_VERTEX_COUNT].mean(axis=0)
    ring = center + (oval - center) * RING_SCALE
    ring[:, 0] = np.clip(ring[:, 0], 0, w - 1)
    ring[:, 1] = np.clip(ring[:, 1], 0, h - 1)

    t = np.linspace(0.0, 1.0, BORDER_POINTS_PER_EDGE)
    border = np.vstack([
        np.column_stack([t * (w - 1), np.zeros_like(t)]),
        np.column_stack([t * (w - 1), np.full_like(t, h - 1)]),
        np.column_stack([np.zeros_like(t), t * (h - 1)]),
        np.column_stack([np.full_like(t, w - 1), t * (h - 1)]),
    ])
    points = np.vstack([ring, border])
    # Evitar duplicados (Delaunay los rechaza)
    _, unique_idx = np.unique(np.round(points, 1), axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def _signed_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def _safe_fraction(src: np.ndarray, disp: np.ndarray, triangles: np.ndarray) -> float:
    """Mayor fracción del desplazamiento (≤ 1) que no pliega triángulos."""
    base = _signed_areas(src, triangles)
    valid = np.abs(base) > 1e-6

    def ok(f: float) -> bool:
        ratio = _signed_areas(src + disp * f, triangles)[valid] / base[valid]
        return bool((ratio > MIN_AREA_RATIO).all())

    if ok(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def _bbox(dst: np.ndarray, src: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    pts = np.vstack([dst, src])
    x0 = int(max(0, np.floor(pts[:, 0].min()) - 2))
    y0 = int(max(0, np.floor(pts[:, 1].min()) - 2))
    x1 = int(min(w, np.ceil(pts[:, 0].max()) + 3))
    y1 = int(min(h, np.ceil(pts[:, 1].max()) + 3))
    return x0, y0, x1, y1
