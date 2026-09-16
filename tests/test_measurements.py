"""
test_measurements.py — Tests para el módulo de mediciones.

Verifica:
    - Calibración manual y automática
    - Cálculos de tercios faciales
    - Cálculos de quintos faciales
    - Análisis de línea media y desviaciones
    - Motor de reglas clínicas
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landmarks import AnthropometricPoint, LandmarkResult
from measurements import (
    ScaleCalibrator,
    CalibrationResult,
    FacialAnalyzer,
    ClinicalRuleEngine,
    AlertSeverity,
    ThirdsResult,
    FifthsResult,
    MidlineResult,
    run_full_analysis,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def ideal_face_landmarks():
    """
    Crear landmarks de una cara 'ideal' con proporciones perfectas.

    Proporciones:
        - Tercios iguales (100px cada uno = 300px total)
        - Quintos iguales (60px cada uno = 300px total)
        - Sin desviación de línea media
    """
    points = {
        # Línea media vertical: X=200, tercios de 100px
        AnthropometricPoint.TRICHION: (200, 50),
        AnthropometricPoint.GLABELLA: (200, 150),
        AnthropometricPoint.NASION: (200, 160),
        AnthropometricPoint.NASAL_BRIDGE: (200, 180),
        AnthropometricPoint.PRONASALE: (200, 220),
        AnthropometricPoint.SUBNASALE: (200, 250),
        AnthropometricPoint.STOMION_SUPERIOR: (200, 280),
        AnthropometricPoint.STOMION_INFERIOR: (200, 290),
        AnthropometricPoint.POGONION: (200, 340),
        AnthropometricPoint.MENTON: (200, 350),

        # Ojos simétricos respecto a X=200
        AnthropometricPoint.ENDOCANTHION_L: (170, 160),
        AnthropometricPoint.ENDOCANTHION_R: (230, 160),
        AnthropometricPoint.EXOCANTHION_L: (110, 160),
        AnthropometricPoint.EXOCANTHION_R: (290, 160),

        # Quintos con orejas
        AnthropometricPoint.TRAGION_L: (50, 160),
        AnthropometricPoint.TRAGION_R: (350, 160),

        # Labios y nariz lateral
        AnthropometricPoint.CHEILION_L: (170, 285),
        AnthropometricPoint.CHEILION_R: (230, 285),
        AnthropometricPoint.ALAR_L: (185, 230),
        AnthropometricPoint.ALAR_R: (215, 230),

        # Perfil
        AnthropometricPoint.SOFT_TISSUE_A: (200, 255),
        AnthropometricPoint.SOFT_TISSUE_B: (200, 310),
    }
    return LandmarkResult(
        points=points,
        raw_landmarks=[(0, 0, 0)] * 478,
        image_width=400,
        image_height=500,
        detection_confidence=0.95,
        estimated_points=[AnthropometricPoint.TRICHION],
    )


@pytest.fixture
def asymmetric_landmarks():
    """Landmarks con asimetrías significativas para testear alertas."""
    points = {
        AnthropometricPoint.TRICHION: (200, 50),
        AnthropometricPoint.GLABELLA: (200, 100),       # Tercio sup: 50px
        AnthropometricPoint.NASION: (200, 120),
        AnthropometricPoint.NASAL_BRIDGE: (200, 140),
        AnthropometricPoint.PRONASALE: (215, 200),       # Desviación nasal de 15px
        AnthropometricPoint.SUBNASALE: (200, 230),       # Tercio medio: 130px
        AnthropometricPoint.STOMION_SUPERIOR: (200, 260),
        AnthropometricPoint.STOMION_INFERIOR: (200, 270),
        AnthropometricPoint.POGONION: (210, 370),        # Desviación mentón de 10px
        AnthropometricPoint.MENTON: (200, 400),          # Tercio inf: 170px

        AnthropometricPoint.ENDOCANTHION_L: (170, 130),
        AnthropometricPoint.ENDOCANTHION_R: (230, 130),
        AnthropometricPoint.EXOCANTHION_L: (130, 130),
        AnthropometricPoint.EXOCANTHION_R: (280, 130),

        AnthropometricPoint.TRAGION_L: (50, 130),
        AnthropometricPoint.TRAGION_R: (350, 130),

        AnthropometricPoint.CHEILION_L: (170, 265),
        AnthropometricPoint.CHEILION_R: (230, 265),
        AnthropometricPoint.ALAR_L: (185, 210),
        AnthropometricPoint.ALAR_R: (215, 210),

        AnthropometricPoint.SOFT_TISSUE_A: (200, 235),
        AnthropometricPoint.SOFT_TISSUE_B: (200, 300),
    }
    return LandmarkResult(
        points=points,
        raw_landmarks=[(0, 0, 0)] * 478,
        image_width=400,
        image_height=500,
        detection_confidence=0.90,
        estimated_points=[AnthropometricPoint.TRICHION],
    )


@pytest.fixture
def calibration_1mm_per_px():
    """Calibración simple: 1 mm = 1 px."""
    return CalibrationResult(
        mm_per_pixel=1.0,
        method="test",
        reference_distance_mm=100.0,
        reference_distance_px=100.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Calibración
# ──────────────────────────────────────────────────────────────────────────────

class TestScaleCalibrator:
    """Tests para la calibración de escala."""

    def test_manual_calibration(self):
        """Verificar calibración manual con distancia conocida."""
        cal = ScaleCalibrator.calibrate_manual(
            p1=(0, 0), p2=(100, 0), real_distance_mm=30.0
        )
        assert abs(cal.mm_per_pixel - 0.3) < 0.001
        assert cal.method == "manual"

    def test_manual_calibration_diagonal(self):
        """Calibración con puntos en diagonal."""
        cal = ScaleCalibrator.calibrate_manual(
            p1=(0, 0), p2=(30, 40), real_distance_mm=50.0
        )
        # Distancia diagonal: sqrt(30² + 40²) = 50 px
        assert abs(cal.mm_per_pixel - 1.0) < 0.001

    def test_manual_calibration_zero_distance_raises(self):
        """Debe lanzar error si la distancia real es 0."""
        with pytest.raises(ValueError):
            ScaleCalibrator.calibrate_manual(
                p1=(0, 0), p2=(100, 0), real_distance_mm=0.0
            )

    def test_manual_calibration_same_points_raises(self):
        """Debe lanzar error si los puntos son idénticos."""
        with pytest.raises(ValueError):
            ScaleCalibrator.calibrate_manual(
                p1=(50, 50), p2=(50, 50), real_distance_mm=10.0
            )

    def test_intercantal_calibration(self, ideal_face_landmarks):
        """Verificar calibración automática por distancia intercantal."""
        cal = ScaleCalibrator.calibrate_intercantal(ideal_face_landmarks)
        # Distancia intercantal: |230 - 170| = 60 px
        # mm_per_pixel = 32.0 / 60.0 ≈ 0.5333
        assert abs(cal.mm_per_pixel - 32.0 / 60.0) < 0.001
        assert cal.method == "intercantal"

    def test_pixel_per_mm_property(self):
        """Verificar propiedad inversa pixel_per_mm."""
        cal = CalibrationResult(
            mm_per_pixel=0.5, method="test",
            reference_distance_mm=50.0, reference_distance_px=100.0,
        )
        assert abs(cal.pixel_per_mm - 2.0) < 0.001


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Tercios Faciales
# ──────────────────────────────────────────────────────────────────────────────

class TestThirds:
    """Tests para el cálculo de tercios faciales."""

    def test_ideal_thirds(self, ideal_face_landmarks, calibration_1mm_per_px):
        """En una cara ideal, los tercios deben ser ~33.3% cada uno."""
        analyzer = FacialAnalyzer()
        result = analyzer.compute_thirds(
            ideal_face_landmarks, calibration_1mm_per_px.mm_per_pixel
        )

        assert result.total_height_mm > 0

        # Cada tercio debe estar cerca del 33.3%
        for seg in [result.upper, result.middle, result.lower]:
            assert abs(seg.percentage - 33.3) < 1.0, (
                f"{seg.name}: {seg.percentage:.1f}% (esperado ~33.3%)"
            )

    def test_thirds_sum_to_total(self, ideal_face_landmarks, calibration_1mm_per_px):
        """La suma de los tercios debe ser igual al total."""
        analyzer = FacialAnalyzer()
        result = analyzer.compute_thirds(
            ideal_face_landmarks, calibration_1mm_per_px.mm_per_pixel
        )

        total = result.upper.height_mm + result.middle.height_mm + result.lower.height_mm
        assert abs(total - result.total_height_mm) < 0.1

    def test_thirds_serialization(self, ideal_face_landmarks, calibration_1mm_per_px):
        """Verificar serialización a diccionario."""
        analyzer = FacialAnalyzer()
        result = analyzer.compute_thirds(
            ideal_face_landmarks, calibration_1mm_per_px.mm_per_pixel
        )
        d = result.to_dict()
        assert "total_height_mm" in d
        assert "upper_third" in d
        assert "middle_third" in d
        assert "lower_third" in d


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Quintos Faciales
# ──────────────────────────────────────────────────────────────────────────────

class TestFifths:
    """Tests para el cálculo de quintos faciales."""

    def test_fifths_count(self, ideal_face_landmarks, calibration_1mm_per_px):
        """Deben generarse exactamente 5 segmentos."""
        analyzer = FacialAnalyzer()
        result = analyzer.compute_fifths(
            ideal_face_landmarks, calibration_1mm_per_px.mm_per_pixel
        )
        assert len(result.segments) == 5

    def test_fifths_sum_to_total(self, ideal_face_landmarks, calibration_1mm_per_px):
        """La suma de los quintos debe ser igual al ancho total."""
        analyzer = FacialAnalyzer()
        result = analyzer.compute_fifths(
            ideal_face_landmarks, calibration_1mm_per_px.mm_per_pixel
        )

        total = sum(s.width_mm for s in result.segments)
        assert abs(total - result.total_width_mm) < 0.1


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Línea Media
# ──────────────────────────────────────────────────────────────────────────────

class TestMidline:
    """Tests para el análisis de línea media."""

    def test_ideal_no_deviation(self, ideal_face_landmarks, calibration_1mm_per_px):
        """En una cara ideal, la desviación nasal y del mentón deben ser ~0."""
        analyzer = FacialAnalyzer()
        result = analyzer.compute_midline(
            ideal_face_landmarks, calibration_1mm_per_px.mm_per_pixel
        )

        assert result.nasal_deviation_mm < 1.0
        assert result.chin_deviation_mm < 1.0

    def test_midline_x_between_endocanthions(self, ideal_face_landmarks, calibration_1mm_per_px):
        """La línea media X debe estar entre los cantos internos."""
        analyzer = FacialAnalyzer()
        result = analyzer.compute_midline(
            ideal_face_landmarks, calibration_1mm_per_px.mm_per_pixel
        )

        en_l = ideal_face_landmarks.get(AnthropometricPoint.ENDOCANTHION_L)[0]
        en_r = ideal_face_landmarks.get(AnthropometricPoint.ENDOCANTHION_R)[0]
        assert en_l <= result.midline_x <= en_r

    def test_asymmetric_deviation(self, asymmetric_landmarks, calibration_1mm_per_px):
        """Detectar desviaciones significativas."""
        analyzer = FacialAnalyzer()
        result = analyzer.compute_midline(
            asymmetric_landmarks, calibration_1mm_per_px.mm_per_pixel
        )

        # Pronasale en X=215, midline ~200, desviación = 15px = 15mm
        assert result.nasal_deviation_mm > 10.0
        assert result.nasal_deviation_side == "derecha"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Motor de Reglas Clínicas
# ──────────────────────────────────────────────────────────────────────────────

class TestClinicalRuleEngine:
    """Tests para el motor de reglas clínicas."""

    def test_no_alerts_on_ideal(self, ideal_face_landmarks, calibration_1mm_per_px):
        """Una cara ideal no debe generar alertas de tipo ALERT."""
        analyzer = FacialAnalyzer()
        mm_px = calibration_1mm_per_px.mm_per_pixel

        thirds = analyzer.compute_thirds(ideal_face_landmarks, mm_px)
        fifths = analyzer.compute_fifths(ideal_face_landmarks, mm_px)
        midline = analyzer.compute_midline(ideal_face_landmarks, mm_px)

        engine = ClinicalRuleEngine()
        alerts = engine.evaluate(thirds=thirds, fifths=fifths, midline=midline)

        severe_alerts = [a for a in alerts if a.severity == AlertSeverity.ALERT]
        assert len(severe_alerts) == 0, (
            f"Cara ideal generó alertas severas: "
            f"{[a.code for a in severe_alerts]}"
        )

    def test_nasal_deviation_alert(self, asymmetric_landmarks, calibration_1mm_per_px):
        """Desviación nasal > 2mm debe generar alerta."""
        analyzer = FacialAnalyzer()
        mm_px = calibration_1mm_per_px.mm_per_pixel
        midline = analyzer.compute_midline(asymmetric_landmarks, mm_px)

        engine = ClinicalRuleEngine()
        alerts = engine.evaluate(midline=midline)

        nasal_alerts = [a for a in alerts if "NASAL" in a.code]
        assert len(nasal_alerts) > 0
        assert any(a.severity == AlertSeverity.ALERT for a in nasal_alerts)

    def test_gingival_exposure_alert(self):
        """Exposición gingival > 3mm debe generar alerta."""
        engine = ClinicalRuleEngine()
        alerts = engine.evaluate(gingival_exposure_mm=4.0)

        assert len(alerts) > 0
        gingival_alerts = [a for a in alerts if "GINGIVAL" in a.code]
        assert len(gingival_alerts) > 0
        assert gingival_alerts[0].severity == AlertSeverity.ALERT

    def test_gingival_no_alert_below_threshold(self):
        """Exposición gingival < 2mm no debe generar alertas."""
        engine = ClinicalRuleEngine()
        alerts = engine.evaluate(gingival_exposure_mm=1.5)
        assert len(alerts) == 0

    def test_alerts_sorted_by_severity(self, asymmetric_landmarks, calibration_1mm_per_px):
        """Las alertas deben estar ordenadas por severidad (ALERT primero)."""
        analyzer = FacialAnalyzer()
        mm_px = calibration_1mm_per_px.mm_per_pixel

        thirds = analyzer.compute_thirds(asymmetric_landmarks, mm_px)
        midline = analyzer.compute_midline(asymmetric_landmarks, mm_px)

        engine = ClinicalRuleEngine()
        alerts = engine.evaluate(thirds=thirds, midline=midline,
                                gingival_exposure_mm=5.0)

        if len(alerts) > 1:
            severity_order = {
                AlertSeverity.ALERT: 0,
                AlertSeverity.WARNING: 1,
                AlertSeverity.INFO: 2,
            }
            for i in range(len(alerts) - 1):
                assert severity_order[alerts[i].severity] <= severity_order[alerts[i+1].severity]

    def test_alert_serialization(self):
        """Verificar serialización de alertas a diccionario."""
        engine = ClinicalRuleEngine()
        alerts = engine.evaluate(gingival_exposure_mm=4.0)
        for alert in alerts:
            d = alert.to_dict()
            assert "code" in d
            assert "severity" in d
            assert "message" in d
            assert "value" in d


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Pipeline Completo
# ──────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    """Tests para el pipeline de análisis completo."""

    def test_full_analysis_frontal(self, ideal_face_landmarks, calibration_1mm_per_px):
        """Ejecutar análisis frontal completo sin errores."""
        report = run_full_analysis(
            ideal_face_landmarks,
            calibration_1mm_per_px,
            view_type="frontal",
        )
        assert report.thirds is not None
        assert report.fifths is not None
        assert report.midline is not None
        assert report.profile is None  # No profile en vista frontal

    def test_full_analysis_profile(self, ideal_face_landmarks, calibration_1mm_per_px):
        """Ejecutar análisis de perfil completo sin errores."""
        report = run_full_analysis(
            ideal_face_landmarks,
            calibration_1mm_per_px,
            view_type="profile",
        )
        assert report.thirds is None
        assert report.profile is not None

    def test_report_serialization(self, ideal_face_landmarks, calibration_1mm_per_px):
        """Verificar que el reporte se serializa correctamente a JSON."""
        report = run_full_analysis(
            ideal_face_landmarks,
            calibration_1mm_per_px,
            view_type="frontal",
        )
        d = report.to_dict()
        assert "calibration" in d
        assert "thirds_analysis" in d
        assert "fifths_analysis" in d
        assert "midline_analysis" in d
        assert "alerts" in d
        assert "alert_count" in d
