"""
landmarks.py — Mapeo de MediaPipe Face Mesh a Puntos Antropométricos Estándar.

Este módulo define el mapeo entre los índices de MediaPipe Face Mesh (468/478 landmarks)
y los puntos antropométricos clínicos utilizados en cefalometría y análisis facial estético.

Puntos soportados:
    - Trichion (Tr), Glabella (G), Nasion (N), Pronasale (Pn), Subnasale (Sn)
    - Stomion Superior (Sts), Stomion Inferior (Sti), Pogonion (Pog'), Menton (Me)
    - Cantos oculares internos/externos (Endocanthion / Exocanthion)
    - Comisuras labiales, alas nasales, tragion (aproximación auricular)

Autor: FacialMetrics Pro
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. Definición de Puntos Antropométricos
# ──────────────────────────────────────────────────────────────────────────────

class AnthropometricPoint(str, Enum):
    """Nombres estándar de puntos antropométricos faciales."""

    # --- Línea media vertical (superior → inferior) ---
    TRICHION = "Tr"           # Nacimiento del pelo (estimado)
    GLABELLA = "G"            # Punto más prominente entre cejas
    NASION = "N"              # Raíz nasal / puente nasal superior
    NASAL_BRIDGE = "NB"       # Puente nasal medio
    PRONASALE = "Pn"          # Punta nasal
    SUBNASALE = "Sn"          # Base nasal / columela
    STOMION_SUPERIOR = "Sts"  # Centro del labio superior
    STOMION_INFERIOR = "Sti"  # Centro del labio inferior
    POGONION = "Pog"          # Punto más anterior del mentón blando
    MENTON = "Me"             # Punto más inferior del mentón

    # --- Ojos ---
    ENDOCANTHION_L = "EnL"    # Canto interno ojo izquierdo
    ENDOCANTHION_R = "EnR"    # Canto interno ojo derecho
    EXOCANTHION_L = "ExL"     # Canto externo ojo izquierdo
    EXOCANTHION_R = "ExR"     # Canto externo ojo derecho

    # --- Labios ---
    CHEILION_L = "ChL"        # Comisura labial izquierda
    CHEILION_R = "ChR"        # Comisura labial derecha

    # --- Nariz (lateral) ---
    ALAR_L = "AlL"            # Ala nasal izquierda
    ALAR_R = "AlR"            # Ala nasal derecha

    # --- Orejas (aproximación) ---
    TRAGION_L = "TrL"         # Tragus izquierdo (aproximación)
    TRAGION_R = "TrR"         # Tragus derecho (aproximación)

    # --- Extras para análisis de perfil ---
    SOFT_TISSUE_A = "A'"      # Punto A blando (base nasal / labio sup)
    SOFT_TISSUE_B = "B'"      # Punto B blando (surco labio-mentón)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Mapeo MediaPipe Face Mesh → Puntos Antropométricos
# ──────────────────────────────────────────────────────────────────────────────

# Índices verificados contra el modelo canónico de MediaPipe Face Mesh (478 pts).
# Los índices mapean a las posiciones más cercanas a los puntos antropométricos
# clínicos. Trichion se estima por extrapolación (no cubierto por el mesh).

MEDIAPIPE_INDEX_MAP: dict[AnthropometricPoint, int | None] = {
    # Línea media
    AnthropometricPoint.TRICHION:          None,   # No disponible → estimado
    AnthropometricPoint.GLABELLA:          9,      # Glabela / entre cejas
    AnthropometricPoint.NASION:            168,    # Puente nasal superior
    AnthropometricPoint.NASAL_BRIDGE:      6,      # Puente nasal medio
    AnthropometricPoint.PRONASALE:         4,      # Punta nasal
    AnthropometricPoint.SUBNASALE:         2,      # Base nasal
    AnthropometricPoint.STOMION_SUPERIOR:  13,     # Centro labio superior
    AnthropometricPoint.STOMION_INFERIOR:  14,     # Centro labio inferior
    AnthropometricPoint.POGONION:          175,    # Mentón anterior
    AnthropometricPoint.MENTON:            152,    # Mentón inferior

    # Ojos
    AnthropometricPoint.ENDOCANTHION_L:    133,    # Canto interno izq
    AnthropometricPoint.ENDOCANTHION_R:    362,    # Canto interno der
    AnthropometricPoint.EXOCANTHION_L:     33,     # Canto externo izq
    AnthropometricPoint.EXOCANTHION_R:     263,    # Canto externo der

    # Labios
    AnthropometricPoint.CHEILION_L:        61,     # Comisura izq
    AnthropometricPoint.CHEILION_R:        291,    # Comisura der

    # Nariz lateral
    AnthropometricPoint.ALAR_L:            129,    # Ala nasal izq
    AnthropometricPoint.ALAR_R:            358,    # Ala nasal der

    # Orejas (aproximación — borde lateral del face mesh)
    AnthropometricPoint.TRAGION_L:         234,    # Tragus izq aprox
    AnthropometricPoint.TRAGION_R:         454,    # Tragus der aprox

    # Perfil
    AnthropometricPoint.SOFT_TISSUE_A:     164,    # Surco nasolabial sup
    AnthropometricPoint.SOFT_TISSUE_B:     18,     # Surco labio-mentón
}


# Grupos funcionales para visualización por color
LANDMARK_GROUPS: dict[str, list[AnthropometricPoint]] = {
    "midline": [
        AnthropometricPoint.TRICHION,
        AnthropometricPoint.GLABELLA,
        AnthropometricPoint.NASION,
        AnthropometricPoint.SUBNASALE,
        AnthropometricPoint.MENTON,
    ],
    "eyes": [
        AnthropometricPoint.ENDOCANTHION_L,
        AnthropometricPoint.ENDOCANTHION_R,
        AnthropometricPoint.EXOCANTHION_L,
        AnthropometricPoint.EXOCANTHION_R,
    ],
    "nose": [
        AnthropometricPoint.NASAL_BRIDGE,
        AnthropometricPoint.PRONASALE,
        AnthropometricPoint.SUBNASALE,
        AnthropometricPoint.ALAR_L,
        AnthropometricPoint.ALAR_R,
    ],
    "lips": [
        AnthropometricPoint.STOMION_SUPERIOR,
        AnthropometricPoint.STOMION_INFERIOR,
        AnthropometricPoint.CHEILION_L,
        AnthropometricPoint.CHEILION_R,
    ],
    "chin": [
        AnthropometricPoint.POGONION,
        AnthropometricPoint.MENTON,
    ],
    "ears": [
        AnthropometricPoint.TRAGION_L,
        AnthropometricPoint.TRAGION_R,
    ],
    "thirds_horizontal": [
        AnthropometricPoint.TRICHION,
        AnthropometricPoint.GLABELLA,
        AnthropometricPoint.SUBNASALE,
        AnthropometricPoint.MENTON,
    ],
    "fifths_vertical": [
        AnthropometricPoint.TRAGION_L,
        AnthropometricPoint.EXOCANTHION_L,
        AnthropometricPoint.ENDOCANTHION_L,
        AnthropometricPoint.ENDOCANTHION_R,
        AnthropometricPoint.EXOCANTHION_R,
        AnthropometricPoint.TRAGION_R,
    ],
}

# Colores por grupo (BGR para OpenCV)
GROUP_COLORS: dict[str, tuple[int, int, int]] = {
    "midline": (255, 200, 50),    # Celeste
    "eyes":    (230, 216, 0),     # Cyan
    "nose":    (80, 200, 80),     # Verde
    "lips":    (180, 105, 255),   # Rosa
    "chin":    (50, 230, 230),    # Amarillo
    "ears":    (200, 160, 120),   # Azul suave
}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Resultado de Detección
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LandmarkResult:
    """Resultado de la detección de landmarks antropométricos."""

    # Coordenadas pixel de cada punto antropométrico
    points: dict[AnthropometricPoint, tuple[int, int]]

    # Landmarks crudos de MediaPipe (478 puntos) como lista de (x, y, z)
    raw_landmarks: list[tuple[float, float, float]]

    # Dimensiones de la imagen procesada
    image_width: int
    image_height: int

    # Confianza de detección
    detection_confidence: float = 0.0

    # Puntos que fueron estimados (no detectados directamente)
    estimated_points: list[AnthropometricPoint] = field(default_factory=list)

    def get(self, point: AnthropometricPoint) -> tuple[int, int]:
        """Obtener coordenadas pixel de un punto antropométrico."""
        return self.points[point]

    def get_normalized(self, point: AnthropometricPoint) -> tuple[float, float]:
        """Obtener coordenadas normalizadas [0, 1] de un punto."""
        px, py = self.points[point]
        return (px / self.image_width, py / self.image_height)

    def distance_px(
        self, p1: AnthropometricPoint, p2: AnthropometricPoint
    ) -> float:
        """Distancia euclidiana en píxeles entre dos puntos."""
        x1, y1 = self.points[p1]
        x2, y2 = self.points[p2]
        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    def midpoint(
        self, p1: AnthropometricPoint, p2: AnthropometricPoint
    ) -> tuple[int, int]:
        """Punto medio entre dos puntos antropométricos."""
        x1, y1 = self.points[p1]
        x2, y2 = self.points[p2]
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def to_dict(self) -> dict:
        """Serializar a diccionario JSON-compatible."""
        return {
            "points": {
                pt.value: {"x": x, "y": y}
                for pt, (x, y) in self.points.items()
            },
            "image_size": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "detection_confidence": self.detection_confidence,
            "estimated_points": [p.value for p in self.estimated_points],
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Detector de Landmarks
# ──────────────────────────────────────────────────────────────────────────────

class FaceLandmarkDetector:
    """
    Detector de puntos antropométricos faciales basado en MediaPipe Face Mesh.

    Uso:
        detector = FaceLandmarkDetector()
        result = detector.detect(image_bgr)
        glabella_px = result.get(AnthropometricPoint.GLABELLA)
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        trichion_extrapolation_ratio: float = 0.15,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Inicializar el detector.

        Args:
            min_detection_confidence: Confianza mínima para detección facial.
            min_tracking_confidence: Confianza mínima para tracking.
            refine_landmarks: Activar modelo de 478 puntos (iris refinado).
            trichion_extrapolation_ratio: Ratio de extrapolación para Trichion
                como fracción de la distancia G-Me (default: 15%).
            model_path: Ruta opcional al archivo .task de MediaPipe.
        """
        self._trichion_ratio = trichion_extrapolation_ratio
        self._min_detection_confidence = min_detection_confidence
        self._use_tasks_api = False
        self._face_mesh = None
        self._task_detector = None

        # Verificar si existe la API legacy mp.solutions
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            # Usar MediaPipe Tasks API (moderna, requerida en Python 3.12+)
            self._use_tasks_api = True
            import os
            import urllib.request
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if model_path is None:
                default_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(default_dir, "face_landmarker.task")

            if not os.path.exists(model_path):
                task_url = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                )
                urllib.request.urlretrieve(task_url, model_path)

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
            )
            self._task_detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> Optional[LandmarkResult]:
        """
        Detectar landmarks antropométricos en una imagen facial.

        Args:
            image: Imagen en formato BGR (OpenCV) o RGB.

        Returns:
            LandmarkResult con coordenadas de todos los puntos, o None si no
            se detecta ningún rostro.
        """
        h, w = image.shape[:2]

        # MediaPipe requiere RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        raw_landmarks: list[tuple[float, float, float]] = []

        if self._use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            task_result = self._task_detector.detect(mp_image)
            if not task_result.face_landmarks:
                return None
            landmarks_proto = task_result.face_landmarks[0]
            for lm in landmarks_proto:
                raw_landmarks.append((lm.x * w, lm.y * h, lm.z * w))
        else:
            results = self._face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                return None
            face_landmarks = results.multi_face_landmarks[0]
            for lm in face_landmarks.landmark:
                raw_landmarks.append((lm.x * w, lm.y * h, lm.z * w))

        # Mapear a puntos antropométricos
        points: dict[AnthropometricPoint, tuple[int, int]] = {}
        estimated: list[AnthropometricPoint] = []

        for point, mp_index in MEDIAPIPE_INDEX_MAP.items():
            if mp_index is not None and mp_index < len(raw_landmarks):
                x, y, _ = raw_landmarks[mp_index]
                points[point] = (int(round(x)), int(round(y)))

        # Estimar Trichion por extrapolación
        trichion = self._estimate_trichion(points, w, h)
        points[AnthropometricPoint.TRICHION] = trichion
        estimated.append(AnthropometricPoint.TRICHION)

        return LandmarkResult(
            points=points,
            raw_landmarks=raw_landmarks,
            image_width=w,
            image_height=h,
            detection_confidence=min_detection_confidence
            if (min_detection_confidence := 0.5) else 0.5,
            estimated_points=estimated,
        )

    def _estimate_trichion(
        self,
        points: dict[AnthropometricPoint, tuple[int, int]],
        img_w: int,
        img_h: int,
    ) -> tuple[int, int]:
        """
        Estimar la posición del Trichion (nacimiento del pelo) por
        extrapolación desde la Glabela.

        Heurística: Trichion está aproximadamente a un 15% de la distancia
        G-Me por encima de la Glabela, sobre el eje medio facial.

        Args:
            points: Puntos ya detectados (necesita G y Me).
            img_w: Ancho de imagen.
            img_h: Alto de imagen.

        Returns:
            Coordenadas (x, y) estimadas del Trichion.
        """
        g = points.get(AnthropometricPoint.GLABELLA)
        me = points.get(AnthropometricPoint.MENTON)

        if g is None or me is None:
            # Fallback: usar la parte más alta de la frente en el mesh (idx 10)
            # Si tampoco existe, usar 10% del alto de imagen
            return (img_w // 2, int(img_h * 0.08))

        # Distancia vertical G → Me
        face_height = abs(me[1] - g[1])

        # Extrapolar hacia arriba
        extrapolation = int(face_height * self._trichion_ratio)

        # Trichion: misma X que Glabela, Y más arriba
        tr_x = g[0]
        tr_y = max(0, g[1] - extrapolation)

        return (tr_x, tr_y)

    def update_landmark(
        self,
        result: LandmarkResult,
        point: AnthropometricPoint,
        new_coords: tuple[int, int],
    ) -> LandmarkResult:
        """
        Actualizar manualmente la posición de un landmark (ej: Trichion
        corregido por el profesional).

        Args:
            result: Resultado de detección original.
            point: Punto a actualizar.
            new_coords: Nuevas coordenadas (x, y) en píxeles.

        Returns:
            LandmarkResult actualizado.
        """
        result.points[point] = new_coords
        if point in result.estimated_points:
            result.estimated_points.remove(point)
        return result

    def close(self) -> None:
        """Liberar recursos de MediaPipe."""
        self._face_mesh.close()

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def get_group_for_point(point: AnthropometricPoint) -> str | None:
    """Obtener el grupo funcional al que pertenece un punto."""
    for group_name, group_points in LANDMARK_GROUPS.items():
        if group_name in ("thirds_horizontal", "fifths_vertical"):
            continue  # Estos son agrupaciones de análisis, no de visualización
        if point in group_points:
            return group_name
    return None


def get_color_for_point(point: AnthropometricPoint) -> tuple[int, int, int]:
    """Obtener color BGR para un punto según su grupo."""
    group = get_group_for_point(point)
    if group and group in GROUP_COLORS:
        return GROUP_COLORS[group]
    return (200, 200, 200)  # Gris por defecto


def get_display_name(point: AnthropometricPoint) -> str:
    """Obtener nombre legible para display en overlays."""
    names: dict[AnthropometricPoint, str] = {
        AnthropometricPoint.TRICHION: "Trichion",
        AnthropometricPoint.GLABELLA: "Glabella",
        AnthropometricPoint.NASION: "Nasion",
        AnthropometricPoint.NASAL_BRIDGE: "Nasal Bridge",
        AnthropometricPoint.PRONASALE: "Pronasale",
        AnthropometricPoint.SUBNASALE: "Subnasale",
        AnthropometricPoint.STOMION_SUPERIOR: "Stomion Sup.",
        AnthropometricPoint.STOMION_INFERIOR: "Stomion Inf.",
        AnthropometricPoint.POGONION: "Pogonion",
        AnthropometricPoint.MENTON: "Menton",
        AnthropometricPoint.ENDOCANTHION_L: "Endocanthion L",
        AnthropometricPoint.ENDOCANTHION_R: "Endocanthion R",
        AnthropometricPoint.EXOCANTHION_L: "Exocanthion L",
        AnthropometricPoint.EXOCANTHION_R: "Exocanthion R",
        AnthropometricPoint.CHEILION_L: "Cheilion L",
        AnthropometricPoint.CHEILION_R: "Cheilion R",
        AnthropometricPoint.ALAR_L: "Alar L",
        AnthropometricPoint.ALAR_R: "Alar R",
        AnthropometricPoint.TRAGION_L: "Tragion L",
        AnthropometricPoint.TRAGION_R: "Tragion R",
        AnthropometricPoint.SOFT_TISSUE_A: "Punto A'",
        AnthropometricPoint.SOFT_TISSUE_B: "Punto B'",
    }
    return names.get(point, point.value)
