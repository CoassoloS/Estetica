"""
test_landmarks.py — Tests para el módulo de landmarks.

Verifica:
    - Definición correcta de todos los puntos antropométricos
    - Mapeo de índices de MediaPipe
    - Estimación de Trichion
    - Funciones de utilidad (get_display_name, get_color_for_point)
"""

import sys
import os
import pytest
import numpy as np

# Añadir directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarks import (
    AnthropometricPoint,
    MEDIAPIPE_INDEX_MAP,
    LANDMARK_GROUPS,
    GROUP_COLORS,
    LandmarkResult,
    FaceLandmarkDetector,
    get_group_for_point,
    get_color_for_point,
    get_display_name,
)


class TestAnthropometricPoints:
    """Tests para la definición de puntos antropométricos."""

    def test_all_points_defined(self):
        """Verificar que todos los puntos esperados están definidos."""
        expected = [
            "Tr", "G", "N", "NB", "Pn", "Sn", "Sts", "Sti",
            "Pog", "Me", "EnL", "EnR", "ExL", "ExR",
            "ChL", "ChR", "AlL", "AlR", "TrL", "TrR",
            "A'", "B'",
        ]
        actual = [p.value for p in AnthropometricPoint]
        for point_value in expected:
            assert point_value in actual, f"Falta punto: {point_value}"

    def test_point_count(self):
        """Verificar número total de puntos."""
        assert len(AnthropometricPoint) == 22


class TestMediaPipeMapping:
    """Tests para el mapeo de índices de MediaPipe."""

    def test_all_points_mapped(self):
        """Verificar que todos los puntos tienen un mapeo definido."""
        for point in AnthropometricPoint:
            assert point in MEDIAPIPE_INDEX_MAP, f"Falta mapeo: {point.value}"

    def test_trichion_is_none(self):
        """Trichion debe ser None (estimado, no disponible en MediaPipe)."""
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.TRICHION] is None

    def test_known_indices(self):
        """Verificar índices conocidos y documentados."""
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.PRONASALE] == 4
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.MENTON] == 152
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.GLABELLA] == 9
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.NASION] == 168
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.SUBNASALE] == 2
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.STOMION_SUPERIOR] == 13
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.STOMION_INFERIOR] == 14

    def test_eye_indices(self):
        """Verificar índices de cantos oculares."""
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.ENDOCANTHION_L] == 133
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.ENDOCANTHION_R] == 362
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.EXOCANTHION_L] == 33
        assert MEDIAPIPE_INDEX_MAP[AnthropometricPoint.EXOCANTHION_R] == 263

    def test_indices_within_range(self):
        """Verificar que todos los índices están dentro del rango 0-477."""
        for point, idx in MEDIAPIPE_INDEX_MAP.items():
            if idx is not None:
                assert 0 <= idx <= 477, (
                    f"{point.value}: índice {idx} fuera de rango [0, 477]"
                )


class TestLandmarkGroups:
    """Tests para los grupos de landmarks."""

    def test_essential_groups_exist(self):
        """Verificar que los grupos esenciales están definidos."""
        essential = ["midline", "eyes", "nose", "lips", "chin", "ears",
                     "thirds_horizontal", "fifths_vertical"]
        for group in essential:
            assert group in LANDMARK_GROUPS, f"Falta grupo: {group}"

    def test_thirds_group_has_four_points(self):
        """El grupo de tercios debe tener 4 puntos (Tr, G, Sn, Me)."""
        thirds = LANDMARK_GROUPS["thirds_horizontal"]
        assert len(thirds) == 4

    def test_fifths_group_has_six_points(self):
        """El grupo de quintos debe tener 6 puntos (6 líneas verticales)."""
        fifths = LANDMARK_GROUPS["fifths_vertical"]
        assert len(fifths) == 6


class TestLandmarkResult:
    """Tests para LandmarkResult."""

    @pytest.fixture
    def sample_result(self):
        """Crear un LandmarkResult de ejemplo con datos sintéticos."""
        points = {
            AnthropometricPoint.GLABELLA: (200, 100),
            AnthropometricPoint.MENTON: (200, 400),
            AnthropometricPoint.TRICHION: (200, 50),
            AnthropometricPoint.NASION: (200, 120),
            AnthropometricPoint.PRONASALE: (203, 200),
            AnthropometricPoint.SUBNASALE: (200, 230),
            AnthropometricPoint.ENDOCANTHION_L: (170, 130),
            AnthropometricPoint.ENDOCANTHION_R: (230, 130),
            AnthropometricPoint.EXOCANTHION_L: (140, 130),
            AnthropometricPoint.EXOCANTHION_R: (260, 130),
            AnthropometricPoint.STOMION_SUPERIOR: (200, 260),
            AnthropometricPoint.STOMION_INFERIOR: (200, 270),
            AnthropometricPoint.POGONION: (201, 370),
        }
        return LandmarkResult(
            points=points,
            raw_landmarks=[(0, 0, 0)] * 478,
            image_width=400,
            image_height=500,
            detection_confidence=0.95,
            estimated_points=[AnthropometricPoint.TRICHION],
        )

    def test_get_point(self, sample_result):
        """Verificar get() retorna coordenadas correctas."""
        assert sample_result.get(AnthropometricPoint.GLABELLA) == (200, 100)

    def test_distance_px(self, sample_result):
        """Verificar cálculo de distancia en píxeles."""
        dist = sample_result.distance_px(
            AnthropometricPoint.ENDOCANTHION_L,
            AnthropometricPoint.ENDOCANTHION_R,
        )
        assert abs(dist - 60.0) < 0.1  # 230-170 = 60 px horizontal

    def test_midpoint(self, sample_result):
        """Verificar cálculo de punto medio."""
        mid = sample_result.midpoint(
            AnthropometricPoint.ENDOCANTHION_L,
            AnthropometricPoint.ENDOCANTHION_R,
        )
        assert mid == (200, 130)

    def test_normalized_coords(self, sample_result):
        """Verificar normalización de coordenadas."""
        nx, ny = sample_result.get_normalized(AnthropometricPoint.GLABELLA)
        assert abs(nx - 0.5) < 0.01   # 200/400 = 0.5
        assert abs(ny - 0.2) < 0.01   # 100/500 = 0.2

    def test_to_dict(self, sample_result):
        """Verificar serialización a diccionario."""
        d = sample_result.to_dict()
        assert "points" in d
        assert "image_size" in d
        assert d["image_size"]["width"] == 400
        assert "estimated_points" in d
        assert "Tr" in d["estimated_points"]


class TestTrichionEstimation:
    """Tests para la estimación de Trichion."""

    def test_estimation_above_glabella(self):
        """Trichion estimado debe estar por encima de Glabella."""
        detector = FaceLandmarkDetector.__new__(FaceLandmarkDetector)
        detector._trichion_ratio = 0.15

        points = {
            AnthropometricPoint.GLABELLA: (200, 200),
            AnthropometricPoint.MENTON: (200, 500),
        }

        tr = detector._estimate_trichion(points, 400, 600)
        assert tr[1] < 200, "Trichion Y debe ser menor que Glabella Y"

    def test_estimation_ratio(self):
        """Verificar que el ratio de extrapolación se aplica correctamente."""
        detector = FaceLandmarkDetector.__new__(FaceLandmarkDetector)
        detector._trichion_ratio = 0.15

        points = {
            AnthropometricPoint.GLABELLA: (200, 200),
            AnthropometricPoint.MENTON: (200, 400),
        }

        tr = detector._estimate_trichion(points, 400, 500)
        # face_height = 400 - 200 = 200
        # extrapolation = 200 * 0.15 = 30
        # tr_y = 200 - 30 = 170
        assert tr == (200, 170)

    def test_estimation_clamps_to_zero(self):
        """Trichion no debe salirse de la imagen (Y >= 0)."""
        detector = FaceLandmarkDetector.__new__(FaceLandmarkDetector)
        detector._trichion_ratio = 0.5  # Ratio muy alto

        points = {
            AnthropometricPoint.GLABELLA: (200, 20),
            AnthropometricPoint.MENTON: (200, 400),
        }

        tr = detector._estimate_trichion(points, 400, 500)
        assert tr[1] >= 0, "Trichion Y no debe ser negativo"


class TestUtilities:
    """Tests para funciones de utilidad."""

    def test_get_display_name(self):
        """Verificar nombres legibles de puntos."""
        assert get_display_name(AnthropometricPoint.GLABELLA) == "Glabella"
        assert get_display_name(AnthropometricPoint.PRONASALE) == "Pronasale"
        assert get_display_name(AnthropometricPoint.MENTON) == "Menton"

    def test_get_color_for_point(self):
        """Verificar que cada punto tiene un color asignado."""
        for point in AnthropometricPoint:
            color = get_color_for_point(point)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_get_group_for_point(self):
        """Verificar asignación de grupo."""
        assert get_group_for_point(AnthropometricPoint.PRONASALE) == "nose"
        assert get_group_for_point(AnthropometricPoint.ENDOCANTHION_L) == "eyes"
        assert get_group_for_point(AnthropometricPoint.STOMION_SUPERIOR) == "lips"
