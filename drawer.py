"""
drawer.py — Renderizado de Overlays Clínicos para Análisis Facial.

Este módulo renderiza sobre las imágenes faciales:
    - Líneas de tercios faciales (horizontales, negro)
    - Líneas de quintos faciales (verticales, rojo)
    - Línea media facial y desviaciones (azul/rojo)
    - Puntos anatómicos con labels
    - Análisis de perfil lateral
    - Zonas semi-transparentes por segmento

Usa OpenCV como motor de renderizado con parámetros configurables.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from landmarks import (
    AnthropometricPoint,
    LandmarkResult,
    get_color_for_point,
    get_display_name,
    LANDMARK_GROUPS,
)
from measurements import (
    ThirdsResult,
    FifthsResult,
    MidlineResult,
    ProfileResult,
    ClinicalReport,
    ClinicalAlert,
    AlertSeverity,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuración Visual
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DrawConfig:
    """Configuración visual para el renderizado de overlays."""

    # Colores (BGR)
    thirds_line_color: tuple[int, int, int] = (40, 40, 40)       # Negro suave
    thirds_zone_colors: list[tuple[int, int, int]] = None        # Por tercio
    fifths_line_color: tuple[int, int, int] = (60, 60, 230)      # Rojo
    fifths_zone_colors: list[tuple[int, int, int]] = None        # Por quinto
    midline_color: tuple[int, int, int] = (230, 160, 50)         # Azul celeste
    deviation_color: tuple[int, int, int] = (50, 50, 230)        # Rojo
    profile_ref_color: tuple[int, int, int] = (80, 200, 80)      # Verde
    profile_proj_color: tuple[int, int, int] = (255, 180, 50)    # Celeste
    landmark_radius: int = 5
    line_thickness: int = 2
    dash_length: int = 12
    label_font_scale: float = 0.45
    label_thickness: int = 1
    overlay_alpha: float = 0.12  # Opacidad de zonas coloreadas
    margin_right: int = 180      # Margen derecho para labels de medidas

    def __post_init__(self):
        if self.thirds_zone_colors is None:
            self.thirds_zone_colors = [
                (255, 220, 150),   # Tercio sup: azul suave
                (150, 255, 200),   # Tercio medio: verde suave
                (180, 180, 255),   # Tercio inf: rosa suave
            ]
        if self.fifths_zone_colors is None:
            self.fifths_zone_colors = [
                (200, 180, 255),   # Q1: violeta suave
                (255, 220, 180),   # Q2: celeste
                (180, 255, 220),   # Q3: verde
                (255, 220, 180),   # Q4: celeste
                (200, 180, 255),   # Q5: violeta suave
            ]


# Configuración por defecto
DEFAULT_CONFIG = DrawConfig()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Utilidades de Dibujo
# ──────────────────────────────────────────────────────────────────────────────

def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
) -> None:
    """Dibujar una línea punteada entre dos puntos."""
    dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    if dist < 1:
        return

    n_dashes = int(dist / dash_length)
    if n_dashes < 1:
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return

    for i in range(0, n_dashes, 2):
        t1 = i / n_dashes
        t2 = min((i + 1) / n_dashes, 1.0)

        x1 = int(pt1[0] + (pt2[0] - pt1[0]) * t1)
        y1 = int(pt1[1] + (pt2[1] - pt1[1]) * t1)
        x2 = int(pt1[0] + (pt2[0] - pt1[0]) * t2)
        y2 = int(pt1[1] + (pt2[1] - pt1[1]) * t2)

        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def _draw_label(
    img: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[tuple[int, int, int]] = (30, 30, 30),
    font_scale: float = 0.45,
    thickness: int = 1,
    padding: int = 4,
) -> None:
    """Dibujar texto con fondo semi-transparente."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    # Fondo
    if bg_color is not None:
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (x - padding, y - th - padding),
            (x + tw + padding, y + baseline + padding),
            bg_color,
            -1,
        )
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Texto
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_measurement_label(
    img: np.ndarray,
    text: str,
    y_pos: int,
    x_start: int,
    color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.5,
) -> None:
    """Dibujar label de medida en el margen derecho."""
    _draw_label(
        img,
        text,
        (x_start + 8, y_pos),
        color=color,
        bg_color=(20, 20, 20),
        font_scale=font_scale,
    )


def _draw_zone(
    img: np.ndarray,
    pt_tl: tuple[int, int],
    pt_br: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 0.12,
) -> None:
    """Dibujar una zona semi-transparente rectangular."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt_tl, pt_br, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_arrow(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    tip_length: float = 0.3,
) -> None:
    """Dibujar una flecha."""
    cv2.arrowedLine(
        img, pt1, pt2, color, thickness, cv2.LINE_AA, tipLength=tip_length
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Dibujo de Puntos Anatómicos
# ──────────────────────────────────────────────────────────────────────────────

def draw_landmarks(
    image: np.ndarray,
    landmarks: LandmarkResult,
    config: DrawConfig = DEFAULT_CONFIG,
    show_labels: bool = True,
    points_filter: Optional[list[AnthropometricPoint]] = None,
) -> np.ndarray:
    """
    Dibujar puntos anatómicos sobre la imagen.

    Args:
        image: Imagen BGR de entrada.
        landmarks: Resultado de detección de landmarks.
        config: Configuración visual.
        show_labels: Mostrar nombres de los puntos.
        points_filter: Si se especifica, solo dibujar estos puntos.

    Returns:
        Imagen con los puntos dibujados.
    """
    img = image.copy()

    for point, (x, y) in landmarks.points.items():
        if points_filter and point not in points_filter:
            continue

        color = get_color_for_point(point)
        is_estimated = point in landmarks.estimated_points

        # Punto: círculo relleno + borde
        cv2.circle(img, (x, y), config.landmark_radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), config.landmark_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Si es estimado, dibujar anillo adicional punteado
        if is_estimated:
            cv2.circle(img, (x, y), config.landmark_radius + 4, (0, 180, 255), 1, cv2.LINE_AA)

        # Label
        if show_labels:
            label = get_display_name(point)
            if is_estimated:
                label += " *"
            _draw_label(
                img,
                label,
                (x + config.landmark_radius + 6, y + 4),
                color=color,
                font_scale=config.label_font_scale,
                thickness=config.label_thickness,
            )

    return img


# ──────────────────────────────────────────────────────────────────────────────
# 4. Dibujo de Tercios Faciales
# ──────────────────────────────────────────────────────────────────────────────

def draw_thirds(
    image: np.ndarray,
    landmarks: LandmarkResult,
    thirds: ThirdsResult,
    config: DrawConfig = DEFAULT_CONFIG,
    show_zones: bool = True,
    show_measurements: bool = True,
) -> np.ndarray:
    """
    Dibujar las líneas de tercios faciales horizontales.

    Dibuja 4 líneas horizontales a nivel de Tr, G, Sn, Me con zonas
    semi-transparentes y medidas en mm en el margen derecho.
    """
    img = image.copy()
    h, w = img.shape[:2]
    AP = AnthropometricPoint

    # Puntos de referencia
    tr = landmarks.get(AP.TRICHION)
    g = landmarks.get(AP.GLABELLA)
    sn = landmarks.get(AP.SUBNASALE)
    me = landmarks.get(AP.MENTON)

    y_points = [tr[1], g[1], sn[1], me[1]]
    labels_right = [
        f"Tr",
        f"G",
        f"Sn",
        f"Me",
    ]

    # Zonas semi-transparentes
    if show_zones:
        x_left = max(0, min(tr[0], g[0], sn[0], me[0]) - 60)
        x_right = min(w, max(tr[0], g[0], sn[0], me[0]) + 60)

        for i in range(3):
            _draw_zone(
                img,
                (x_left, y_points[i]),
                (x_right, y_points[i + 1]),
                config.thirds_zone_colors[i],
                config.overlay_alpha,
            )

    # Líneas horizontales punteadas
    for i, y in enumerate(y_points):
        _draw_dashed_line(
            img,
            (0, y),
            (w, y),
            config.thirds_line_color,
            config.line_thickness,
            config.dash_length,
        )

        # Label del punto en el margen izquierdo
        _draw_label(
            img,
            labels_right[i],
            (8, y - 6),
            color=(220, 220, 220),
            font_scale=0.4,
        )

    # Medidas en mm (bracket derecho)
    if show_measurements:
        margin_x = w - config.margin_right

        segments_data = [
            (thirds.upper, y_points[0], y_points[1]),
            (thirds.middle, y_points[1], y_points[2]),
            (thirds.lower, y_points[2], y_points[3]),
        ]

        for seg, y_top, y_bottom in segments_data:
            y_mid = (y_top + y_bottom) // 2
            bracket_x = margin_x + 10

            # Bracket vertical
            cv2.line(img, (bracket_x, y_top + 2), (bracket_x, y_bottom - 2),
                     (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(img, (bracket_x - 4, y_top + 2), (bracket_x + 4, y_top + 2),
                     (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(img, (bracket_x - 4, y_bottom - 2), (bracket_x + 4, y_bottom - 2),
                     (200, 200, 200), 1, cv2.LINE_AA)

            # Texto con medida
            _draw_measurement_label(
                img,
                f"{seg.height_mm:.1f}mm ({seg.percentage:.0f}%)",
                y_mid + 5,
                bracket_x + 6,
                color=(255, 255, 255),
                font_scale=0.42,
            )

        # Subdivisión del tercio inferior
        sts = landmarks.get(AP.STOMION_SUPERIOR)
        sti = landmarks.get(AP.STOMION_INFERIOR)

        # Línea fina punteada a nivel de Sts
        _draw_dashed_line(
            img,
            (w // 4, sts[1]),
            (3 * w // 4, sts[1]),
            (100, 100, 100),
            1,
            6,
        )

        # Label subdivisión
        sub_x = margin_x - 60
        _draw_label(
            img,
            f"Labio sup: {thirds.lower_upper_lip_mm:.1f}mm",
            (sub_x, (sn[1] + sts[1]) // 2 + 4),
            color=(200, 200, 255),
            font_scale=0.35,
        )
        _draw_label(
            img,
            f"Mentón: {thirds.lower_chin_mm:.1f}mm",
            (sub_x, (sti[1] + me[1]) // 2 + 4),
            color=(200, 200, 255),
            font_scale=0.35,
        )

    return img


# ──────────────────────────────────────────────────────────────────────────────
# 5. Dibujo de Quintos Faciales
# ──────────────────────────────────────────────────────────────────────────────

def draw_fifths(
    image: np.ndarray,
    landmarks: LandmarkResult,
    fifths: FifthsResult,
    config: DrawConfig = DEFAULT_CONFIG,
    show_zones: bool = True,
    show_measurements: bool = True,
) -> np.ndarray:
    """
    Dibujar las líneas de quintos faciales verticales.

    Dibuja 6 líneas verticales rojas a nivel de Tragion L/R, Exocanthion L/R,
    Endocanthion L/R, con zonas coloreadas y medidas en mm.
    """
    img = image.copy()
    h, w = img.shape[:2]
    AP = AnthropometricPoint

    # 6 puntos de referencia (izq → der)
    ref_points = [
        (AP.TRAGION_L, "TrL"),
        (AP.EXOCANTHION_L, "ExL"),
        (AP.ENDOCANTHION_L, "EnL"),
        (AP.ENDOCANTHION_R, "EnR"),
        (AP.EXOCANTHION_R, "ExR"),
        (AP.TRAGION_R, "TrR"),
    ]

    x_coords = [landmarks.get(pt)[0] for pt, _ in ref_points]

    # Zona vertical de referencia (entre ojos aprox)
    y_top = landmarks.get(AP.EXOCANTHION_L)[1] - 40
    y_bottom = landmarks.get(AP.SUBNASALE)[1] + 20

    # Zonas semi-transparentes
    if show_zones:
        for i in range(5):
            _draw_zone(
                img,
                (x_coords[i], y_top),
                (x_coords[i + 1], y_bottom),
                config.fifths_zone_colors[i],
                config.overlay_alpha * 1.2,
            )

    # Líneas verticales
    for i, (pt, label) in enumerate(ref_points):
        x = x_coords[i]
        _draw_dashed_line(
            img,
            (x, 0),
            (x, h),
            config.fifths_line_color,
            config.line_thickness,
            config.dash_length,
        )

        # Label superior
        _draw_label(
            img,
            label,
            (x - 10, 20),
            color=(180, 180, 255),
            font_scale=0.38,
        )

    # Medidas de ancho
    if show_measurements:
        y_measure = y_bottom + 30

        for i, seg in enumerate(fifths.segments):
            x_mid = (x_coords[i] + x_coords[i + 1]) // 2

            # Bracket horizontal
            cv2.line(img, (x_coords[i] + 2, y_measure),
                     (x_coords[i + 1] - 2, y_measure),
                     (200, 200, 200), 1, cv2.LINE_AA)

            # Ticks
            cv2.line(img, (x_coords[i] + 2, y_measure - 3),
                     (x_coords[i] + 2, y_measure + 3),
                     (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(img, (x_coords[i + 1] - 2, y_measure - 3),
                     (x_coords[i + 1] - 2, y_measure + 3),
                     (200, 200, 200), 1, cv2.LINE_AA)

            # Medida
            _draw_label(
                img,
                f"{seg.width_mm:.1f}mm",
                (x_mid - 15, y_measure + 18),
                color=(255, 255, 255),
                font_scale=0.35,
            )
            _draw_label(
                img,
                f"({seg.percentage:.0f}%)",
                (x_mid - 12, y_measure + 34),
                color=(180, 180, 180),
                font_scale=0.3,
            )

    return img


# ──────────────────────────────────────────────────────────────────────────────
# 6. Dibujo de Línea Media y Desviaciones
# ──────────────────────────────────────────────────────────────────────────────

def draw_midline(
    image: np.ndarray,
    landmarks: LandmarkResult,
    midline: MidlineResult,
    config: DrawConfig = DEFAULT_CONFIG,
    show_deviations: bool = True,
) -> np.ndarray:
    """
    Dibujar la línea media facial y desviaciones de nariz y mentón.
    """
    img = image.copy()
    h, w = img.shape[:2]
    AP = AnthropometricPoint

    # Línea media (azul celeste, continua)
    cv2.line(
        img,
        (midline.midline_x, 0),
        (midline.midline_x, h),
        config.midline_color,
        config.line_thickness,
        cv2.LINE_AA,
    )

    # Label "Línea Media"
    _draw_label(
        img,
        "Linea Media",
        (midline.midline_x - 30, h - 20),
        color=config.midline_color,
        font_scale=0.35,
    )

    if show_deviations:
        pn = landmarks.get(AP.PRONASALE)
        pog = landmarks.get(AP.POGONION)

        # Desviación nasal
        if midline.nasal_deviation_mm > 0.5:
            # Línea horizontal roja: midline → pronasale
            cv2.line(
                img,
                (midline.midline_x, pn[1]),
                (pn[0], pn[1]),
                config.deviation_color,
                2,
                cv2.LINE_AA,
            )

            # Flecha y label
            _draw_label(
                img,
                f"Desv. nasal: {midline.nasal_deviation_mm:.1f}mm {midline.nasal_deviation_side}",
                (min(midline.midline_x, pn[0]) - 10, pn[1] - 12),
                color=config.deviation_color,
                font_scale=0.38,
            )

        # Desviación del mentón
        if midline.chin_deviation_mm > 0.5:
            cv2.line(
                img,
                (midline.midline_x, pog[1]),
                (pog[0], pog[1]),
                config.deviation_color,
                2,
                cv2.LINE_AA,
            )

            _draw_label(
                img,
                f"Desv. menton: {midline.chin_deviation_mm:.1f}mm {midline.chin_deviation_side}",
                (min(midline.midline_x, pog[0]) - 10, pog[1] + 18),
                color=config.deviation_color,
                font_scale=0.38,
            )

    return img


# ──────────────────────────────────────────────────────────────────────────────
# 7. Dibujo de Análisis de Perfil
# ──────────────────────────────────────────────────────────────────────────────

def draw_profile(
    image: np.ndarray,
    landmarks: LandmarkResult,
    profile: ProfileResult,
    config: DrawConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Dibujar el análisis de perfil lateral con línea de referencia estética
    y proyecciones anteroposteriores.
    """
    img = image.copy()
    h, w = img.shape[:2]
    AP = AnthropometricPoint

    sn = landmarks.get(AP.SUBNASALE)
    pn = landmarks.get(AP.PRONASALE)
    sts = landmarks.get(AP.STOMION_SUPERIOR)
    sti = landmarks.get(AP.STOMION_INFERIOR)
    pog = landmarks.get(AP.POGONION)

    # Línea de referencia vertical desde Sn (verde)
    cv2.line(
        img,
        (sn[0], 0),
        (sn[0], h),
        config.profile_ref_color,
        config.line_thickness,
        cv2.LINE_AA,
    )

    _draw_label(
        img,
        "Ref. Sn",
        (sn[0] + 5, 25),
        color=config.profile_ref_color,
        font_scale=0.38,
    )

    # Proyecciones como líneas horizontales
    projections = [
        ("Nariz", pn, profile.nasal_projection_mm),
        ("Labio Sup", sts, profile.upper_lip_projection_mm),
        ("Labio Inf", sti, profile.lower_lip_projection_mm),
        ("Menton", pog, profile.chin_projection_mm),
    ]

    for name, point, proj_mm in projections:
        # Línea horizontal desde ref hasta el punto
        color = config.profile_proj_color
        cv2.line(
            img,
            (sn[0], point[1]),
            (point[0], point[1]),
            color,
            2,
            cv2.LINE_AA,
        )

        # Punto
        cv2.circle(img, point, 4, color, -1, cv2.LINE_AA)

        # Label
        sign = "+" if proj_mm >= 0 else ""
        label_x = max(sn[0], point[0]) + 8
        _draw_label(
            img,
            f"{name}: {sign}{proj_mm:.1f}mm",
            (label_x, point[1] + 4),
            color=color,
            font_scale=0.4,
        )

    # Ángulo nasolabial
    _draw_label(
        img,
        f"Ang. nasolabial: {profile.nasolabial_angle:.1f} grados",
        (20, h - 40),
        color=(220, 220, 220),
        font_scale=0.42,
    )

    return img


# ──────────────────────────────────────────────────────────────────────────────
# 8. Dibujo de Alertas Clínicas
# ──────────────────────────────────────────────────────────────────────────────

def draw_alerts_panel(
    image: np.ndarray,
    alerts: list[ClinicalAlert],
    position: str = "bottom",
) -> np.ndarray:
    """
    Dibujar un panel de alertas clínicas en la imagen.

    Args:
        image: Imagen BGR.
        alerts: Lista de alertas clínicas.
        position: "bottom" o "right" — posición del panel.

    Returns:
        Imagen con panel de alertas.
    """
    if not alerts:
        return image.copy()

    h, w = image.shape[:2]
    line_height = 22
    panel_height = max(60, len(alerts) * line_height + 30)

    # Crear imagen expandida con panel inferior
    if position == "bottom":
        result = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
        result[:h, :w] = image
        # Fondo del panel
        result[h:, :] = (25, 25, 25)
        # Separador
        cv2.line(result, (0, h), (w, h), (80, 80, 80), 2)

        # Título
        _draw_label(
            result,
            "ALERTAS CLINICAS",
            (10, h + 18),
            color=(255, 255, 255),
            bg_color=None,
            font_scale=0.5,
            thickness=1,
        )

        # Alertas
        y_offset = h + 38
        for alert in alerts:
            severity_icon = {
                AlertSeverity.INFO: "[OK]",
                AlertSeverity.WARNING: "[!!]",
                AlertSeverity.ALERT: "[XX]",
            }
            severity_color = {
                AlertSeverity.INFO: (100, 200, 100),      # Verde
                AlertSeverity.WARNING: (50, 200, 255),     # Amarillo
                AlertSeverity.ALERT: (80, 80, 255),        # Rojo
            }

            icon = severity_icon[alert.severity]
            color = severity_color[alert.severity]

            _draw_label(
                result,
                f"{icon} {alert.message}",
                (15, y_offset),
                color=color,
                bg_color=None,
                font_scale=0.38,
            )
            y_offset += line_height

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 9. Pipeline de Renderizado Completo
# ──────────────────────────────────────────────────────────────────────────────

def render_full_analysis(
    image: np.ndarray,
    landmarks: LandmarkResult,
    report: ClinicalReport,
    config: DrawConfig = DEFAULT_CONFIG,
    layers: Optional[dict[str, bool]] = None,
) -> np.ndarray:
    """
    Renderizar el análisis facial completo con todas las capas de overlay.

    Args:
        image: Imagen BGR original.
        landmarks: Resultado de detección de landmarks.
        report: Reporte clínico con métricas calculadas.
        config: Configuración visual.
        layers: Diccionario de capas a mostrar:
            - "landmarks": Puntos anatómicos (default: True)
            - "thirds": Tercios faciales (default: True)
            - "fifths": Quintos faciales (default: True)
            - "midline": Línea media y desviaciones (default: True)
            - "profile": Análisis de perfil (default: True)
            - "alerts": Panel de alertas (default: True)

    Returns:
        Imagen con todos los overlays renderizados.
    """
    if layers is None:
        layers = {
            "landmarks": True,
            "thirds": True,
            "fifths": True,
            "midline": True,
            "profile": True,
            "alerts": True,
        }

    result = image.copy()

    # Capa de tercios
    if layers.get("thirds", False) and report.thirds:
        result = draw_thirds(result, landmarks, report.thirds, config)

    # Capa de quintos
    if layers.get("fifths", False) and report.fifths:
        result = draw_fifths(result, landmarks, report.fifths, config)

    # Capa de línea media
    if layers.get("midline", False) and report.midline:
        result = draw_midline(result, landmarks, report.midline, config)

    # Capa de perfil
    if layers.get("profile", False) and report.profile:
        result = draw_profile(result, landmarks, report.profile, config)

    # Capa de puntos anatómicos (siempre al final para que estén encima)
    if layers.get("landmarks", False):
        result = draw_landmarks(result, landmarks, config)

    # Panel de alertas
    if layers.get("alerts", False) and report.alerts:
        result = draw_alerts_panel(result, report.alerts)

    return result


def render_comparison(
    original: np.ndarray,
    annotated: np.ndarray,
    label_left: str = "Original",
    label_right: str = "Analisis",
) -> np.ndarray:
    """
    Crear imagen de comparación side-by-side (Original vs Analizado).

    Args:
        original: Imagen original sin anotaciones.
        annotated: Imagen con overlays de análisis.
        label_left: Label para imagen izquierda.
        label_right: Label para imagen derecha.

    Returns:
        Imagen combinada con separador vertical.
    """
    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]

    # Igualar alturas
    target_h = max(h1, h2)
    if h1 != target_h:
        scale = target_h / h1
        original = cv2.resize(original, (int(w1 * scale), target_h))
        w1 = original.shape[1]
    if h2 != target_h:
        scale = target_h / h2
        annotated = cv2.resize(annotated, (int(w2 * scale), target_h))
        w2 = annotated.shape[1]

    # Separador
    sep_width = 4
    result = np.zeros((target_h, w1 + sep_width + w2, 3), dtype=np.uint8)
    result[:, :w1] = original
    result[:, w1:w1 + sep_width] = (80, 80, 80)
    result[:, w1 + sep_width:] = annotated

    # Labels
    _draw_label(result, label_left, (10, 25), color=(255, 255, 255), font_scale=0.55)
    _draw_label(result, label_right, (w1 + sep_width + 10, 25), color=(255, 255, 255), font_scale=0.55)

    return result
