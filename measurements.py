"""
measurements.py — Calibración Métrica y Cálculos Antropométricos Faciales.

Este módulo implementa:
    1. Calibración de escala (píxeles → milímetros)
    2. Cálculo de tercios faciales (ley de tercios)
    3. Cálculo de quintos faciales (ley de quintos)
    4. Análisis de línea media y desviaciones
    5. Análisis de perfil lateral
    6. Motor de reglas clínicas con alertas diagnósticas

Autor: FacialMetrics Pro
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from landmarks import AnthropometricPoint, LandmarkResult


# ──────────────────────────────────────────────────────────────────────────────
# 1. Calibración de Escala
# ──────────────────────────────────────────────────────────────────────────────

# Constantes antropométricas estándar (adulto promedio)
INTERCANTAL_DISTANCE_MM = 32.0       # Distancia intercantal media (adulto)
INTERPUPILAR_DISTANCE_MM = 63.0      # Distancia interpupilar media


@dataclass
class CalibrationResult:
    """Resultado de calibración píxel → milímetro."""
    mm_per_pixel: float
    method: str               # "manual" | "intercantal" | "interpupilar"
    reference_distance_mm: float
    reference_distance_px: float

    @property
    def pixel_per_mm(self) -> float:
        return 1.0 / self.mm_per_pixel if self.mm_per_pixel > 0 else 0.0


class ScaleCalibrator:
    """
    Calibrador de escala para convertir distancias en píxeles a milímetros.

    Soporta dos modos:
        1. Manual: El usuario selecciona dos puntos de una regla visible en la
           foto e ingresa la distancia real en mm.
        2. Automático (fallback): Usa la distancia intercantal promedio (~32 mm)
           como referencia de escala.
    """

    @staticmethod
    def calibrate_manual(
        p1: tuple[int, int],
        p2: tuple[int, int],
        real_distance_mm: float,
    ) -> CalibrationResult:
        """
        Calibración manual usando dos puntos de referencia conocidos.

        Args:
            p1: Coordenadas pixel del primer punto de la regla.
            p2: Coordenadas pixel del segundo punto de la regla.
            real_distance_mm: Distancia real entre los puntos en milímetros.

        Returns:
            CalibrationResult con el factor mm_per_pixel.

        Raises:
            ValueError: Si la distancia real es <= 0 o los puntos coinciden.
        """
        if real_distance_mm <= 0:
            raise ValueError(
                f"La distancia real debe ser > 0, recibido: {real_distance_mm}"
            )

        dist_px = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        if dist_px < 1e-6:
            raise ValueError("Los dos puntos de calibración son idénticos.")

        mm_per_pixel = real_distance_mm / dist_px

        return CalibrationResult(
            mm_per_pixel=mm_per_pixel,
            method="manual",
            reference_distance_mm=real_distance_mm,
            reference_distance_px=dist_px,
        )

    @staticmethod
    def calibrate_intercantal(
        landmarks: LandmarkResult,
        reference_mm: float = INTERCANTAL_DISTANCE_MM,
    ) -> CalibrationResult:
        """
        Calibración automática usando la distancia intercantal como referencia.

        Usa la distancia entre Endocanthion L y Endocanthion R, asumiendo
        un valor promedio de ~32 mm para adultos.

        Args:
            landmarks: Resultado de detección de landmarks.
            reference_mm: Distancia intercantal de referencia en mm.

        Returns:
            CalibrationResult con el factor mm_per_pixel.
        """
        dist_px = landmarks.distance_px(
            AnthropometricPoint.ENDOCANTHION_L,
            AnthropometricPoint.ENDOCANTHION_R,
        )

        if dist_px < 1e-6:
            raise ValueError(
                "No se pudo calcular la distancia intercantal (puntos "
                "coincidentes o no detectados)."
            )

        mm_per_pixel = reference_mm / dist_px

        return CalibrationResult(
            mm_per_pixel=mm_per_pixel,
            method="intercantal",
            reference_distance_mm=reference_mm,
            reference_distance_px=dist_px,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Resultados de Análisis (Dataclasses)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThirdSegment:
    """Segmento individual de un tercio facial."""
    name: str
    top_point: str
    bottom_point: str
    height_px: float
    height_mm: float
    percentage: float          # Porcentaje del total facial
    ideal_percentage: float    # Porcentaje ideal (33.3%)
    deviation: float           # Desviación respecto al ideal (%)


@dataclass
class ThirdsResult:
    """Resultado del análisis de tercios faciales."""
    upper: ThirdSegment        # Tr → G
    middle: ThirdSegment       # G → Sn
    lower: ThirdSegment        # Sn → Me
    total_height_px: float
    total_height_mm: float

    # Subdivisión del tercio inferior
    lower_upper_lip_mm: float    # Sn → Sts (1/3 del tercio inferior ideal)
    lower_chin_mm: float         # Sti → Me  (2/3 del tercio inferior ideal)
    lower_lip_ratio: float       # Ratio real labio/mentón

    def to_dict(self) -> dict:
        return {
            "total_height_mm": round(self.total_height_mm, 1),
            "upper_third": {
                "height_mm": round(self.upper.height_mm, 1),
                "percentage": round(self.upper.percentage, 1),
                "deviation": round(self.upper.deviation, 1),
            },
            "middle_third": {
                "height_mm": round(self.middle.height_mm, 1),
                "percentage": round(self.middle.percentage, 1),
                "deviation": round(self.middle.deviation, 1),
            },
            "lower_third": {
                "height_mm": round(self.lower.height_mm, 1),
                "percentage": round(self.lower.percentage, 1),
                "deviation": round(self.lower.deviation, 1),
                "upper_lip_mm": round(self.lower_upper_lip_mm, 1),
                "lower_chin_mm": round(self.lower_chin_mm, 1),
                "lip_chin_ratio": round(self.lower_lip_ratio, 2),
            },
        }


@dataclass
class FifthSegment:
    """Segmento individual de un quinto facial."""
    name: str
    left_point: str
    right_point: str
    width_px: float
    width_mm: float
    percentage: float
    ideal_percentage: float     # 20% cada uno
    deviation: float


@dataclass
class FifthsResult:
    """Resultado del análisis de quintos faciales."""
    segments: list[FifthSegment]  # 5 segmentos de izquierda a derecha
    total_width_px: float
    total_width_mm: float

    def to_dict(self) -> dict:
        return {
            "total_width_mm": round(self.total_width_mm, 1),
            "fifths": [
                {
                    "name": s.name,
                    "width_mm": round(s.width_mm, 1),
                    "percentage": round(s.percentage, 1),
                    "deviation": round(s.deviation, 1),
                }
                for s in self.segments
            ],
        }


@dataclass
class MidlineResult:
    """Resultado del análisis de línea media facial."""
    midline_x: int                   # Coordenada X de la línea media
    midline_top: tuple[int, int]     # Punto superior de la línea media
    midline_bottom: tuple[int, int]  # Punto inferior de la línea media

    nasal_deviation_px: float        # Desviación nasal en px
    nasal_deviation_mm: float        # Desviación nasal en mm
    nasal_deviation_side: str        # "izquierda" | "derecha" | "centrado"

    chin_deviation_px: float         # Desviación del mentón en px
    chin_deviation_mm: float         # Desviación del mentón en mm
    chin_deviation_side: str         # "izquierda" | "derecha" | "centrado"

    def to_dict(self) -> dict:
        return {
            "nasal_deviation_mm": round(self.nasal_deviation_mm, 1),
            "nasal_deviation_side": self.nasal_deviation_side,
            "chin_deviation_mm": round(self.chin_deviation_mm, 1),
            "chin_deviation_side": self.chin_deviation_side,
        }


@dataclass
class ProfileResult:
    """Resultado del análisis de perfil lateral."""
    # Proyecciones anteroposteriores respecto a la línea de referencia (mm)
    nasal_projection_mm: float       # Pronasale respecto a vertical Sn
    upper_lip_projection_mm: float   # Labio superior respecto a línea estética
    lower_lip_projection_mm: float   # Labio inferior respecto a línea estética
    chin_projection_mm: float        # Pogonion respecto a línea estética

    # Ángulo nasolabial (Sn-Pn con respecto a vertical)
    nasolabial_angle: float          # En grados

    # Línea de referencia estética usada
    reference_line: str              # "ricketts" | "steiner" | "subnasal_vertical"

    def to_dict(self) -> dict:
        return {
            "nasal_projection_mm": round(self.nasal_projection_mm, 1),
            "upper_lip_projection_mm": round(self.upper_lip_projection_mm, 1),
            "lower_lip_projection_mm": round(self.lower_lip_projection_mm, 1),
            "chin_projection_mm": round(self.chin_projection_mm, 1),
            "nasolabial_angle": round(self.nasolabial_angle, 1),
            "reference_line": self.reference_line,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Alertas Clínicas
# ──────────────────────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    """Nivel de severidad de una alerta clínica."""
    INFO = "info"           # 🟢 Informativo
    WARNING = "warning"     # 🟡 Atención
    ALERT = "alert"         # 🔴 Requiere evaluación


@dataclass
class ClinicalAlert:
    """Alerta clínica individual."""
    code: str                 # Código identificador (ej: "NASAL_DEVIATION")
    severity: AlertSeverity
    category: str             # "nasal" | "labial" | "mentoniano" | "gingival" | "simetria"
    message: str              # Descripción legible
    value: float              # Valor medido
    threshold: float          # Umbral que disparó la alerta
    unit: str = "mm"          # Unidad de la medida
    recommendation: str = ""  # Sugerencia clínica

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "value": round(self.value, 1),
            "threshold": self.threshold,
            "unit": self.unit,
            "recommendation": self.recommendation,
        }


@dataclass
class ClinicalReport:
    """Reporte clínico completo con métricas y alertas."""
    thirds: Optional[ThirdsResult] = None
    fifths: Optional[FifthsResult] = None
    midline: Optional[MidlineResult] = None
    profile: Optional[ProfileResult] = None
    alerts: list[ClinicalAlert] = field(default_factory=list)
    calibration: Optional[CalibrationResult] = None

    def to_dict(self) -> dict:
        result: dict = {}
        if self.calibration:
            result["calibration"] = {
                "mm_per_pixel": round(self.calibration.mm_per_pixel, 4),
                "method": self.calibration.method,
            }
        if self.thirds:
            result["thirds_analysis"] = self.thirds.to_dict()
        if self.fifths:
            result["fifths_analysis"] = self.fifths.to_dict()
        if self.midline:
            result["midline_analysis"] = self.midline.to_dict()
        if self.profile:
            result["profile_analysis"] = self.profile.to_dict()
        result["alerts"] = [a.to_dict() for a in self.alerts]
        result["alert_count"] = {
            "info": sum(
                1 for a in self.alerts if a.severity == AlertSeverity.INFO
            ),
            "warning": sum(
                1 for a in self.alerts if a.severity == AlertSeverity.WARNING
            ),
            "alert": sum(
                1 for a in self.alerts if a.severity == AlertSeverity.ALERT
            ),
        }
        return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. Analizador Facial
# ──────────────────────────────────────────────────────────────────────────────

class FacialAnalyzer:
    """
    Motor de cálculos antropométricos faciales.

    Calcula tercios, quintos, línea media, desviaciones y métricas de perfil.
    """

    @staticmethod
    def _dist(
        p1: tuple[int, int], p2: tuple[int, int]
    ) -> float:
        """Distancia euclidiana entre dos puntos."""
        return float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))

    @staticmethod
    def _vertical_dist(
        p1: tuple[int, int], p2: tuple[int, int]
    ) -> float:
        """Distancia vertical (absoluta) entre dos puntos."""
        return abs(p2[1] - p1[1])

    @staticmethod
    def _horizontal_dist(
        p1: tuple[int, int], p2: tuple[int, int]
    ) -> float:
        """Distancia horizontal (absoluta) entre dos puntos."""
        return abs(p2[0] - p1[0])

    # ── Tercios Faciales ─────────────────────────────────────────────────

    def compute_thirds(
        self,
        landmarks: LandmarkResult,
        mm_per_pixel: float,
    ) -> ThirdsResult:
        """
        Calcular los tercios faciales (Ley de Tercios).

        Tercios:
            - Superior: Trichion (Tr) → Glabella (G)
            - Medio:    Glabella (G)  → Subnasale (Sn)
            - Inferior: Subnasale (Sn) → Menton (Me)

        Subdivisión del tercio inferior:
            - Labio superior: Sn → Sts (idealmente 1/3)
            - Labio inferior + mentón: Sti → Me (idealmente 2/3)
        """
        AP = AnthropometricPoint

        tr = landmarks.get(AP.TRICHION)
        g = landmarks.get(AP.GLABELLA)
        sn = landmarks.get(AP.SUBNASALE)
        me = landmarks.get(AP.MENTON)
        sts = landmarks.get(AP.STOMION_SUPERIOR)
        sti = landmarks.get(AP.STOMION_INFERIOR)

        # Alturas en píxeles (distancia vertical)
        upper_px = self._vertical_dist(tr, g)
        middle_px = self._vertical_dist(g, sn)
        lower_px = self._vertical_dist(sn, me)
        total_px = upper_px + middle_px + lower_px

        # Conversión a mm
        upper_mm = upper_px * mm_per_pixel
        middle_mm = middle_px * mm_per_pixel
        lower_mm = lower_px * mm_per_pixel
        total_mm = total_px * mm_per_pixel

        # Porcentajes
        upper_pct = (upper_px / total_px * 100) if total_px > 0 else 0
        middle_pct = (middle_px / total_px * 100) if total_px > 0 else 0
        lower_pct = (lower_px / total_px * 100) if total_px > 0 else 0

        ideal = 100.0 / 3.0  # 33.33%

        # Subdivisión del tercio inferior
        lip_upper_px = self._vertical_dist(sn, sts)
        chin_px = self._vertical_dist(sti, me)
        lip_upper_mm = lip_upper_px * mm_per_pixel
        chin_mm = chin_px * mm_per_pixel
        lip_ratio = (lip_upper_px / chin_px) if chin_px > 0 else 0

        return ThirdsResult(
            upper=ThirdSegment(
                name="Tercio Superior",
                top_point="Tr",
                bottom_point="G",
                height_px=upper_px,
                height_mm=upper_mm,
                percentage=upper_pct,
                ideal_percentage=ideal,
                deviation=upper_pct - ideal,
            ),
            middle=ThirdSegment(
                name="Tercio Medio",
                top_point="G",
                bottom_point="Sn",
                height_px=middle_px,
                height_mm=middle_mm,
                percentage=middle_pct,
                ideal_percentage=ideal,
                deviation=middle_pct - ideal,
            ),
            lower=ThirdSegment(
                name="Tercio Inferior",
                top_point="Sn",
                bottom_point="Me",
                height_px=lower_px,
                height_mm=lower_mm,
                percentage=lower_pct,
                ideal_percentage=ideal,
                deviation=lower_pct - ideal,
            ),
            total_height_px=total_px,
            total_height_mm=total_mm,
            lower_upper_lip_mm=lip_upper_mm,
            lower_chin_mm=chin_mm,
            lower_lip_ratio=lip_ratio,
        )

    # ── Quintos Faciales ─────────────────────────────────────────────────

    def compute_fifths(
        self,
        landmarks: LandmarkResult,
        mm_per_pixel: float,
    ) -> FifthsResult:
        """
        Calcular los quintos faciales (Ley de Quintos).

        5 segmentos horizontales (izquierda → derecha):
            1. Tragion L → Exocanthion L (oreja-ojo)
            2. Exocanthion L → Endocanthion L (ojo izquierdo)
            3. Endocanthion L → Endocanthion R (puente nasal / intercantal)
            4. Endocanthion R → Exocanthion R (ojo derecho)
            5. Exocanthion R → Tragion R (ojo-oreja)

        En el rostro ideal, cada quinto ≈ 20% del ancho total.
        """
        AP = AnthropometricPoint

        # Puntos de referencia (6 líneas verticales)
        points_sequence = [
            ("Tragion L", AP.TRAGION_L),
            ("Exocanthion L", AP.EXOCANTHION_L),
            ("Endocanthion L", AP.ENDOCANTHION_L),
            ("Endocanthion R", AP.ENDOCANTHION_R),
            ("Exocanthion R", AP.EXOCANTHION_R),
            ("Tragion R", AP.TRAGION_R),
        ]

        fifth_names = [
            "1° Quinto (Oreja L)",
            "2° Quinto (Ojo L)",
            "3° Quinto (Intercantal)",
            "4° Quinto (Ojo R)",
            "5° Quinto (Oreja R)",
        ]

        # Obtener coordenadas X
        xs = [landmarks.get(pt)[0] for _, pt in points_sequence]

        # Ancho total
        total_px = abs(xs[-1] - xs[0])
        total_mm = total_px * mm_per_pixel

        ideal = 20.0  # Cada quinto = 20% del total

        segments: list[FifthSegment] = []
        for i in range(5):
            width_px = abs(xs[i + 1] - xs[i])
            width_mm = width_px * mm_per_pixel
            pct = (width_px / total_px * 100) if total_px > 0 else 0

            segments.append(FifthSegment(
                name=fifth_names[i],
                left_point=points_sequence[i][0],
                right_point=points_sequence[i + 1][0],
                width_px=width_px,
                width_mm=width_mm,
                percentage=pct,
                ideal_percentage=ideal,
                deviation=pct - ideal,
            ))

        return FifthsResult(
            segments=segments,
            total_width_px=total_px,
            total_width_mm=total_mm,
        )

    # ── Línea Media y Desviaciones ───────────────────────────────────────

    def compute_midline(
        self,
        landmarks: LandmarkResult,
        mm_per_pixel: float,
    ) -> MidlineResult:
        """
        Calcular la línea media facial y desviaciones de nariz y mentón.

        La línea media se define como el eje vertical que pasa por el punto
        medio entre los cantos internos de los ojos (Endocanthion L y R).
        """
        AP = AnthropometricPoint

        # Línea media: punto medio entre endocanthions
        en_l = landmarks.get(AP.ENDOCANTHION_L)
        en_r = landmarks.get(AP.ENDOCANTHION_R)
        midline_x = (en_l[0] + en_r[0]) // 2

        # Puntos superior e inferior de la línea media
        g = landmarks.get(AP.GLABELLA)
        me = landmarks.get(AP.MENTON)
        midline_top = (midline_x, g[1])
        midline_bottom = (midline_x, me[1])

        # Desviación nasal
        pn = landmarks.get(AP.PRONASALE)
        nasal_dev_px = pn[0] - midline_x
        nasal_dev_mm = abs(nasal_dev_px) * mm_per_pixel
        nasal_side = (
            "derecha" if nasal_dev_px > 2
            else "izquierda" if nasal_dev_px < -2
            else "centrado"
        )

        # Desviación del mentón
        pog = landmarks.get(AP.POGONION)
        chin_dev_px = pog[0] - midline_x
        chin_dev_mm = abs(chin_dev_px) * mm_per_pixel
        chin_side = (
            "derecha" if chin_dev_px > 2
            else "izquierda" if chin_dev_px < -2
            else "centrado"
        )

        return MidlineResult(
            midline_x=midline_x,
            midline_top=midline_top,
            midline_bottom=midline_bottom,
            nasal_deviation_px=abs(nasal_dev_px),
            nasal_deviation_mm=nasal_dev_mm,
            nasal_deviation_side=nasal_side,
            chin_deviation_px=abs(chin_dev_px),
            chin_deviation_mm=chin_dev_mm,
            chin_deviation_side=chin_side,
        )

    # ── Análisis de Perfil ───────────────────────────────────────────────

    def compute_profile(
        self,
        landmarks: LandmarkResult,
        mm_per_pixel: float,
    ) -> ProfileResult:
        """
        Análisis de perfil lateral.

        Utiliza una línea de referencia vertical trazada desde Subnasale (Sn)
        perpendicular al plano de Frankfort. Mide proyecciones anteroposteriores
        de nariz, labios y mentón respecto a esta línea.

        Nota: Para perfil lateral estricto (vista de 90°), la coordenada X
        representa la proyección anteroposterior.
        """
        AP = AnthropometricPoint

        sn = landmarks.get(AP.SUBNASALE)
        pn = landmarks.get(AP.PRONASALE)
        sts = landmarks.get(AP.STOMION_SUPERIOR)
        sti = landmarks.get(AP.STOMION_INFERIOR)
        pog = landmarks.get(AP.POGONION)

        # La línea de referencia es la vertical que pasa por Sn
        ref_x = sn[0]

        # Proyecciones: positivo = anterior a la línea, negativo = posterior
        nasal_proj_px = pn[0] - ref_x
        upper_lip_proj_px = sts[0] - ref_x
        lower_lip_proj_px = sti[0] - ref_x
        chin_proj_px = pog[0] - ref_x

        nasal_proj_mm = nasal_proj_px * mm_per_pixel
        upper_lip_proj_mm = upper_lip_proj_px * mm_per_pixel
        lower_lip_proj_mm = lower_lip_proj_px * mm_per_pixel
        chin_proj_mm = chin_proj_px * mm_per_pixel

        # Ángulo nasolabial: ángulo entre vector Sn→Pn y la vertical
        # En perfil lateral: el ángulo entre la línea columela-labio y vertical
        dx = pn[0] - sn[0]
        dy = sn[1] - pn[1]  # Y invertido (imagen: Y crece hacia abajo)
        nasolabial_angle = float(np.degrees(np.arctan2(abs(dx), dy)))

        return ProfileResult(
            nasal_projection_mm=nasal_proj_mm,
            upper_lip_projection_mm=upper_lip_proj_mm,
            lower_lip_projection_mm=lower_lip_proj_mm,
            chin_projection_mm=chin_proj_mm,
            nasolabial_angle=nasolabial_angle,
            reference_line="subnasal_vertical",
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Motor de Reglas Clínicas
# ──────────────────────────────────────────────────────────────────────────────

class ClinicalRuleEngine:
    """
    Motor de reglas clínicas para generar alertas diagnósticas basadas en
    los cálculos antropométricos.

    Evalúa métricas contra umbrales clínicos definidos y genera alertas
    con nivel de severidad y recomendaciones.
    """

    # Umbrales clínicos (configurables)
    NASAL_DEVIATION_WARNING_MM = 1.5
    NASAL_DEVIATION_ALERT_MM = 2.0
    CHIN_DEVIATION_WARNING_MM = 2.0
    CHIN_DEVIATION_ALERT_MM = 3.0
    THIRDS_DEVIATION_WARNING_PCT = 10.0
    THIRDS_DEVIATION_ALERT_PCT = 15.0
    LIP_RATIO_IDEAL = 0.5       # Ideal: labio sup = 1/2 del labio inf+mentón
    LIP_RATIO_TOLERANCE = 0.20  # ±20%
    GINGIVAL_EXPOSURE_WARNING_MM = 2.0
    GINGIVAL_EXPOSURE_ALERT_MM = 3.0
    FIFTHS_DEVIATION_WARNING_PCT = 8.0
    FIFTHS_DEVIATION_ALERT_PCT = 15.0

    def evaluate(
        self,
        thirds: Optional[ThirdsResult] = None,
        fifths: Optional[FifthsResult] = None,
        midline: Optional[MidlineResult] = None,
        profile: Optional[ProfileResult] = None,
        gingival_exposure_mm: Optional[float] = None,
    ) -> list[ClinicalAlert]:
        """
        Evaluar todas las métricas y generar alertas clínicas.

        Args:
            thirds: Resultado de análisis de tercios.
            fifths: Resultado de análisis de quintos.
            midline: Resultado de análisis de línea media.
            profile: Resultado de análisis de perfil.
            gingival_exposure_mm: Exposición gingival en sonrisa (mm), si disponible.

        Returns:
            Lista de alertas clínicas ordenadas por severidad.
        """
        alerts: list[ClinicalAlert] = []

        if midline:
            alerts.extend(self._evaluate_midline(midline))

        if thirds:
            alerts.extend(self._evaluate_thirds(thirds))

        if fifths:
            alerts.extend(self._evaluate_fifths(fifths))

        if profile:
            alerts.extend(self._evaluate_profile(profile))

        if gingival_exposure_mm is not None:
            alerts.extend(
                self._evaluate_gingival_exposure(gingival_exposure_mm)
            )

        # Ordenar por severidad (alert > warning > info)
        severity_order = {
            AlertSeverity.ALERT: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.INFO: 2,
        }
        alerts.sort(key=lambda a: severity_order[a.severity])

        return alerts

    def _evaluate_midline(self, midline: MidlineResult) -> list[ClinicalAlert]:
        """Evaluar desviaciones de línea media."""
        alerts: list[ClinicalAlert] = []

        # Desviación nasal
        if midline.nasal_deviation_mm >= self.NASAL_DEVIATION_ALERT_MM:
            alerts.append(ClinicalAlert(
                code="NASAL_DEVIATION",
                severity=AlertSeverity.ALERT,
                category="nasal",
                message=(
                    f"Asimetría nasal significativa: desviación de "
                    f"{midline.nasal_deviation_mm:.1f} mm hacia la "
                    f"{midline.nasal_deviation_side}"
                ),
                value=midline.nasal_deviation_mm,
                threshold=self.NASAL_DEVIATION_ALERT_MM,
                recommendation=(
                    "Evaluar Rinoplastia correctiva o relleno de "
                    "compensación contralateral"
                ),
            ))
        elif midline.nasal_deviation_mm >= self.NASAL_DEVIATION_WARNING_MM:
            alerts.append(ClinicalAlert(
                code="NASAL_DEVIATION_MILD",
                severity=AlertSeverity.WARNING,
                category="nasal",
                message=(
                    f"Desviación nasal leve: {midline.nasal_deviation_mm:.1f} mm "
                    f"hacia la {midline.nasal_deviation_side}"
                ),
                value=midline.nasal_deviation_mm,
                threshold=self.NASAL_DEVIATION_WARNING_MM,
                recommendation=(
                    "Monitorizar. Considerar valoración si es sintomático "
                    "o preocupación estética"
                ),
            ))

        # Desviación del mentón
        if midline.chin_deviation_mm >= self.CHIN_DEVIATION_ALERT_MM:
            alerts.append(ClinicalAlert(
                code="CHIN_DEVIATION",
                severity=AlertSeverity.ALERT,
                category="mentoniano",
                message=(
                    f"Asimetría mentoniana: desviación de "
                    f"{midline.chin_deviation_mm:.1f} mm hacia la "
                    f"{midline.chin_deviation_side}"
                ),
                value=midline.chin_deviation_mm,
                threshold=self.CHIN_DEVIATION_ALERT_MM,
                recommendation="Evaluar mentoplastia o cirugía ortognática",
            ))
        elif midline.chin_deviation_mm >= self.CHIN_DEVIATION_WARNING_MM:
            alerts.append(ClinicalAlert(
                code="CHIN_DEVIATION_MILD",
                severity=AlertSeverity.WARNING,
                category="mentoniano",
                message=(
                    f"Desviación mentoniana leve: "
                    f"{midline.chin_deviation_mm:.1f} mm hacia la "
                    f"{midline.chin_deviation_side}"
                ),
                value=midline.chin_deviation_mm,
                threshold=self.CHIN_DEVIATION_WARNING_MM,
                recommendation="Monitorizar. Evaluar correlación oclusal",
            ))

        return alerts

    def _evaluate_thirds(self, thirds: ThirdsResult) -> list[ClinicalAlert]:
        """Evaluar proporciones de tercios faciales."""
        alerts: list[ClinicalAlert] = []

        for segment in [thirds.upper, thirds.middle, thirds.lower]:
            abs_dev = abs(segment.deviation)

            if abs_dev >= self.THIRDS_DEVIATION_ALERT_PCT:
                direction = "aumentado" if segment.deviation > 0 else "disminuido"
                alerts.append(ClinicalAlert(
                    code=f"THIRDS_IMBALANCE_{segment.name.upper().replace(' ', '_')}",
                    severity=AlertSeverity.ALERT,
                    category="proporciones",
                    message=(
                        f"{segment.name} {direction}: "
                        f"{segment.percentage:.1f}% (ideal: 33.3%)"
                    ),
                    value=segment.percentage,
                    threshold=33.3,
                    unit="%",
                    recommendation=self._thirds_recommendation(
                        segment.name, segment.deviation
                    ),
                ))
            elif abs_dev >= self.THIRDS_DEVIATION_WARNING_PCT:
                direction = "aumentado" if segment.deviation > 0 else "disminuido"
                alerts.append(ClinicalAlert(
                    code=f"THIRDS_DEVIATION_{segment.name.upper().replace(' ', '_')}",
                    severity=AlertSeverity.WARNING,
                    category="proporciones",
                    message=(
                        f"{segment.name} levemente {direction}: "
                        f"{segment.percentage:.1f}% (ideal: 33.3%)"
                    ),
                    value=segment.percentage,
                    threshold=33.3,
                    unit="%",
                ))

        # Ratio labio/mentón en tercio inferior
        ideal_ratio = self.LIP_RATIO_IDEAL
        tolerance = self.LIP_RATIO_TOLERANCE
        if thirds.lower_lip_ratio > 0:
            ratio_dev = abs(thirds.lower_lip_ratio - ideal_ratio) / ideal_ratio
            if ratio_dev > tolerance:
                alerts.append(ClinicalAlert(
                    code="LIP_CHIN_RATIO",
                    severity=AlertSeverity.WARNING,
                    category="labial",
                    message=(
                        f"Ratio labio superior/mentón alterado: "
                        f"{thirds.lower_lip_ratio:.2f} (ideal: ~0.50)"
                    ),
                    value=thirds.lower_lip_ratio,
                    threshold=ideal_ratio,
                    unit="ratio",
                    recommendation="Evaluar queiloplastia o mentoplastia",
                ))

        return alerts

    def _evaluate_fifths(self, fifths: FifthsResult) -> list[ClinicalAlert]:
        """Evaluar proporciones de quintos faciales."""
        alerts: list[ClinicalAlert] = []

        for segment in fifths.segments:
            abs_dev = abs(segment.deviation)

            if abs_dev >= self.FIFTHS_DEVIATION_ALERT_PCT:
                alerts.append(ClinicalAlert(
                    code=f"FIFTHS_IMBALANCE_{segment.name[:10].upper().replace(' ', '_')}",
                    severity=AlertSeverity.WARNING,
                    category="proporciones",
                    message=(
                        f"{segment.name} desproporcionado: "
                        f"{segment.percentage:.1f}% (ideal: 20%)"
                    ),
                    value=segment.percentage,
                    threshold=20.0,
                    unit="%",
                ))

        # Simetría: comparar quintos simétricos (1 vs 5, 2 vs 4)
        if len(fifths.segments) == 5:
            asym_ear = abs(
                fifths.segments[0].width_mm - fifths.segments[4].width_mm
            )
            asym_eye = abs(
                fifths.segments[1].width_mm - fifths.segments[3].width_mm
            )

            if asym_ear > 3.0:
                alerts.append(ClinicalAlert(
                    code="FIFTHS_ASYMMETRY_LATERAL",
                    severity=AlertSeverity.INFO,
                    category="simetria",
                    message=(
                        f"Asimetría lateral en quintos: diferencia de "
                        f"{asym_ear:.1f} mm entre segmentos laterales"
                    ),
                    value=asym_ear,
                    threshold=3.0,
                ))

            if asym_eye > 2.0:
                alerts.append(ClinicalAlert(
                    code="FIFTHS_ASYMMETRY_ORBITAL",
                    severity=AlertSeverity.WARNING,
                    category="simetria",
                    message=(
                        f"Asimetría orbital en quintos: diferencia de "
                        f"{asym_eye:.1f} mm entre segmentos oculares"
                    ),
                    value=asym_eye,
                    threshold=2.0,
                ))

        return alerts

    def _evaluate_profile(self, profile: ProfileResult) -> list[ClinicalAlert]:
        """Evaluar métricas de perfil."""
        alerts: list[ClinicalAlert] = []

        # Ángulo nasolabial (normal: 90-110°)
        if profile.nasolabial_angle < 90:
            alerts.append(ClinicalAlert(
                code="NASOLABIAL_ACUTE",
                severity=AlertSeverity.WARNING,
                category="nasal",
                message=(
                    f"Ángulo nasolabial agudo: {profile.nasolabial_angle:.1f}° "
                    f"(normal: 90-110°)"
                ),
                value=profile.nasolabial_angle,
                threshold=90.0,
                unit="°",
                recommendation="Evaluar proyección nasal excesiva o retrusión labial",
            ))
        elif profile.nasolabial_angle > 110:
            alerts.append(ClinicalAlert(
                code="NASOLABIAL_OBTUSE",
                severity=AlertSeverity.WARNING,
                category="nasal",
                message=(
                    f"Ángulo nasolabial obtuso: {profile.nasolabial_angle:.1f}° "
                    f"(normal: 90-110°)"
                ),
                value=profile.nasolabial_angle,
                threshold=110.0,
                unit="°",
                recommendation="Evaluar hipoproyección nasal o protrusión labial",
            ))

        # Proyección del mentón
        if profile.chin_projection_mm < -5:
            alerts.append(ClinicalAlert(
                code="CHIN_RETRUDED",
                severity=AlertSeverity.ALERT,
                category="mentoniano",
                message=(
                    f"Mentón retruido: {profile.chin_projection_mm:.1f} mm "
                    f"posterior a la línea de referencia"
                ),
                value=abs(profile.chin_projection_mm),
                threshold=5.0,
                recommendation=(
                    "Evaluar mentoplastia de avance o relleno de mentón"
                ),
            ))

        return alerts

    def _evaluate_gingival_exposure(
        self, exposure_mm: float
    ) -> list[ClinicalAlert]:
        """Evaluar exposición gingival en sonrisa."""
        alerts: list[ClinicalAlert] = []

        if exposure_mm >= self.GINGIVAL_EXPOSURE_ALERT_MM:
            alerts.append(ClinicalAlert(
                code="GINGIVAL_EXPOSURE",
                severity=AlertSeverity.ALERT,
                category="gingival",
                message=(
                    f"Sonrisa gingival: exposición de {exposure_mm:.1f} mm "
                    f"(umbral: {self.GINGIVAL_EXPOSURE_ALERT_MM} mm)"
                ),
                value=exposure_mm,
                threshold=self.GINGIVAL_EXPOSURE_ALERT_MM,
                recommendation=(
                    "Evaluar toxina botulínica en labio superior, "
                    "gingivoplastia o reposición labial"
                ),
            ))
        elif exposure_mm >= self.GINGIVAL_EXPOSURE_WARNING_MM:
            alerts.append(ClinicalAlert(
                code="GINGIVAL_EXPOSURE_MILD",
                severity=AlertSeverity.WARNING,
                category="gingival",
                message=(
                    f"Exposición gingival leve en sonrisa: {exposure_mm:.1f} mm"
                ),
                value=exposure_mm,
                threshold=self.GINGIVAL_EXPOSURE_WARNING_MM,
                recommendation="Monitorizar. Considerar tratamiento si es preocupación estética",
            ))

        return alerts

    @staticmethod
    def _thirds_recommendation(segment_name: str, deviation: float) -> str:
        """Generar recomendación específica según el tercio afectado."""
        if "Superior" in segment_name:
            if deviation > 0:
                return "Tercio superior aumentado. Considerar valoración de línea de pelo"
            return "Tercio superior disminuido. Variante normal frecuente"
        elif "Medio" in segment_name:
            if deviation > 0:
                return "Tercio medio aumentado. Evaluar dimensión vertical nasal"
            return "Tercio medio disminuido. Evaluar hipoplasia maxilar"
        else:  # Inferior
            if deviation > 0:
                return (
                    "Tercio inferior aumentado. Evaluar exceso vertical "
                    "maxilar, mentoplastia de reducción"
                )
            return (
                "Tercio inferior disminuido. Evaluar hipoplasia mandibular, "
                "mentoplastia de avance"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Pipeline de Análisis Completo
# ──────────────────────────────────────────────────────────────────────────────

def run_full_analysis(
    landmarks: LandmarkResult,
    calibration: CalibrationResult,
    view_type: str = "frontal",
    gingival_exposure_mm: Optional[float] = None,
) -> ClinicalReport:
    """
    Ejecutar el pipeline completo de análisis facial.

    Args:
        landmarks: Resultado de detección de landmarks.
        calibration: Resultado de calibración de escala.
        view_type: Tipo de vista — "frontal" o "profile".
        gingival_exposure_mm: Exposición gingival en sonrisa (mm), si aplica.

    Returns:
        ClinicalReport con todas las métricas y alertas.
    """
    analyzer = FacialAnalyzer()
    rule_engine = ClinicalRuleEngine()
    mm_px = calibration.mm_per_pixel

    report = ClinicalReport(calibration=calibration)

    if view_type == "frontal":
        report.thirds = analyzer.compute_thirds(landmarks, mm_px)
        report.fifths = analyzer.compute_fifths(landmarks, mm_px)
        report.midline = analyzer.compute_midline(landmarks, mm_px)
    elif view_type == "profile":
        report.profile = analyzer.compute_profile(landmarks, mm_px)

    report.alerts = rule_engine.evaluate(
        thirds=report.thirds,
        fifths=report.fifths,
        midline=report.midline,
        profile=report.profile,
        gingival_exposure_mm=gingival_exposure_mm,
    )

    return report
