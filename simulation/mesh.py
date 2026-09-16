"""
simulation/mesh.py — Malla facial 3D a partir de los 478 landmarks de MediaPipe.

La malla usa las coordenadas crudas de MediaPipe (x, y en píxeles; z en la
misma escala que x, negativo = más cerca de la cámara) y la teselación
canónica del Face Mesh. Es la base geométrica común para:

    - el warp 2D de la foto,
    - la re-medición antropométrica,
    - el visor 3D,
    - la verificación de la salida de IA.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from landmarks import AnthropometricPoint, LandmarkResult, MEDIAPIPE_INDEX_MAP


# Los índices 468-477 son el iris refinado: no forman parte de la teselación
FACE_VERTEX_COUNT = 468

# Contorno del óvalo facial (orden perimetral)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]


@lru_cache(maxsize=1)
def tessellation_triangles() -> np.ndarray:
    """
    Triángulos (N, 3) de la teselación canónica de MediaPipe.

    MediaPipe publica la teselación como lista de aristas; los triángulos
    se reconstruyen buscando los ciclos de 3 aristas.
    """
    from mediapipe.tasks.python import vision

    adjacency: dict[int, set[int]] = defaultdict(set)
    for conn in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
        adjacency[conn.start].add(conn.end)
        adjacency[conn.end].add(conn.start)

    triangles: set[tuple[int, int, int]] = set()
    for a, neighbors in adjacency.items():
        for b in neighbors:
            for c in neighbors & adjacency[b]:
                triangles.add(tuple(sorted((a, b, c))))

    return np.array(sorted(triangles), dtype=np.int32)


@dataclass
class FaceMesh3D:
    """Malla facial 3D en coordenadas de imagen (px)."""

    vertices: np.ndarray        # (478, 3) float64 — x, y, z
    image_width: int
    image_height: int

    @classmethod
    def from_landmarks(cls, landmarks: LandmarkResult) -> "FaceMesh3D":
        return cls(
            vertices=np.asarray(landmarks.raw_landmarks, dtype=np.float64),
            image_width=landmarks.image_width,
            image_height=landmarks.image_height,
        )

    @property
    def xy(self) -> np.ndarray:
        return self.vertices[:, :2]

    def displaced(self, displacement: np.ndarray) -> "FaceMesh3D":
        """Nueva malla con los vértices desplazados (N, 3)."""
        return FaceMesh3D(
            vertices=self.vertices + displacement,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    # ── Marco anatómico ──────────────────────────────────────────────────

    def roll_angle(self) -> float:
        """Inclinación de la cabeza en el plano de la imagen (rad), por eje ocular."""
        ex_l = self.vertices[MEDIAPIPE_INDEX_MAP[AnthropometricPoint.EXOCANTHION_L]]
        ex_r = self.vertices[MEDIAPIPE_INDEX_MAP[AnthropometricPoint.EXOCANTHION_R]]
        return float(np.arctan2(ex_r[1] - ex_l[1], ex_r[0] - ex_l[0]))

    def midline_x(self) -> float:
        """X de la línea media facial (en marco sin inclinación)."""
        return float(self.to_face_frame(self.vertices)[[168, 6, 2, 152], 0].mean())

    def to_face_frame(self, points: np.ndarray) -> np.ndarray:
        """Rotar puntos (N, 3) para anular la inclinación de la cabeza."""
        return _rotate_xy(points, -self.roll_angle(), self._pivot())

    def from_face_frame_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Rotar vectores (N, 3) del marco facial al marco de la imagen."""
        return _rotate_xy(vectors, self.roll_angle(), np.zeros(3))

    def _pivot(self) -> np.ndarray:
        return self.vertices[MEDIAPIPE_INDEX_MAP[AnthropometricPoint.NASAL_BRIDGE]]

    # ── Conversión a LandmarkResult ──────────────────────────────────────

    def to_landmark_result(self, reference: LandmarkResult) -> LandmarkResult:
        """
        LandmarkResult frontal a partir de la malla, para re-medir.

        Los puntos estimados (Trichion) se trasladan igual que la Glabela,
        porque no forman parte de la malla.
        """
        points = _points_from_vertices(self.vertices[:, :2], reference)
        return LandmarkResult(
            points=points,
            raw_landmarks=[tuple(v) for v in self.vertices],
            image_width=self.image_width,
            image_height=self.image_height,
            detection_confidence=reference.detection_confidence,
            estimated_points=list(reference.estimated_points),
        )

    def to_profile_landmark_result(self, reference: LandmarkResult) -> LandmarkResult:
        """
        LandmarkResult de perfil sintético: X = proyección anterior (-z).

        Permite usar `FacialAnalyzer.compute_profile` sobre la malla. Las
        proyecciones absolutas dependen de la profundidad estimada por
        MediaPipe; los deltas antes/después son los desplazamientos aplicados.
        """
        frame = self.to_face_frame(self.vertices)
        profile_xy = np.column_stack([-frame[:, 2], frame[:, 1]])
        points = _points_from_vertices(profile_xy, reference, translate_estimated=False)
        g = points[AnthropometricPoint.GLABELLA]
        points[AnthropometricPoint.TRICHION] = g
        return LandmarkResult(
            points=points,
            raw_landmarks=[],
            image_width=self.image_width,
            image_height=self.image_height,
            detection_confidence=reference.detection_confidence,
        )


def _points_from_vertices(
    xy: np.ndarray,
    reference: LandmarkResult,
    translate_estimated: bool = True,
) -> dict[AnthropometricPoint, tuple[int, int]]:
    points: dict[AnthropometricPoint, tuple[int, int]] = {}
    for point, index in MEDIAPIPE_INDEX_MAP.items():
        if index is not None:
            points[point] = (int(round(xy[index, 0])), int(round(xy[index, 1])))

    if translate_estimated:
        g_ref = reference.get(AnthropometricPoint.GLABELLA)
        g_new = points[AnthropometricPoint.GLABELLA]
        for point in reference.estimated_points:
            x, y = reference.get(point)
            points[point] = (x + g_new[0] - g_ref[0], y + g_new[1] - g_ref[1])
    return points


def _rotate_xy(points: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    out = points.astype(np.float64).copy()
    x = points[:, 0] - pivot[0]
    y = points[:, 1] - pivot[1]
    out[:, 0] = pivot[0] + c * x - s * y
    out[:, 1] = pivot[1] + s * x + c * y
    return out
