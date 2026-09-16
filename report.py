"""
report.py — Generación de Reportes PDF Médicos Profesionales.

Genera un PDF estructurado con:
    - Portada con datos del paciente y fecha
    - Imágenes anotadas (frontal y perfil)
    - Tabla de mediciones con rangos normales
    - Alertas clínicas con semáforo de severidad
    - Disclaimer médico

Usa ReportLab para el renderizado del PDF.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from measurements import (
    AlertSeverity,
    ClinicalAlert,
    ClinicalReport,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Estilos Personalizados
# ──────────────────────────────────────────────────────────────────────────────

def _get_styles() -> dict[str, ParagraphStyle]:
    """Generar estilos tipográficos para el reporte."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=22,
            spaceAfter=6 * mm,
            textColor=colors.HexColor("#1a237e"),
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Heading2"],
            fontSize=14,
            spaceAfter=4 * mm,
            spaceBefore=6 * mm,
            textColor=colors.HexColor("#283593"),
            fontName="Helvetica-Bold",
        ),
        "heading": ParagraphStyle(
            "ReportHeading",
            parent=base["Heading3"],
            fontSize=11,
            spaceAfter=3 * mm,
            spaceBefore=4 * mm,
            textColor=colors.HexColor("#1565c0"),
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["Normal"],
            fontSize=9,
            spaceAfter=2 * mm,
            leading=13,
            fontName="Helvetica",
        ),
        "body_bold": ParagraphStyle(
            "ReportBodyBold",
            parent=base["Normal"],
            fontSize=9,
            spaceAfter=2 * mm,
            leading=13,
            fontName="Helvetica-Bold",
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["Normal"],
            fontSize=7,
            textColor=colors.HexColor("#666666"),
            fontName="Helvetica",
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["Normal"],
            fontSize=7,
            textColor=colors.HexColor("#999999"),
            fontName="Helvetica-Oblique",
            alignment=TA_JUSTIFY,
            leading=10,
        ),
        "alert_red": ParagraphStyle(
            "AlertRed",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#c62828"),
            fontName="Helvetica-Bold",
        ),
        "alert_yellow": ParagraphStyle(
            "AlertYellow",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#e65100"),
            fontName="Helvetica-Bold",
        ),
        "alert_green": ParagraphStyle(
            "AlertGreen",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#2e7d32"),
            fontName="Helvetica",
        ),
    }
    return styles


# ──────────────────────────────────────────────────────────────────────────────
# 2. Conversión de Imagen para ReportLab
# ──────────────────────────────────────────────────────────────────────────────

def _cv2_to_reportlab_image(
    img: np.ndarray,
    max_width: float = 170 * mm,
    max_height: float = 120 * mm,
) -> Image:
    """Convertir imagen OpenCV (BGR) a objeto Image de ReportLab."""
    # BGR → RGB → PNG en buffer
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Error al codificar imagen para PDF")

    img_buffer = io.BytesIO(buffer.tobytes())
    h, w = img.shape[:2]

    # Calcular dimensiones manteniendo aspecto
    aspect = w / h
    if w / max_width > h / max_height:
        display_width = max_width
        display_height = max_width / aspect
    else:
        display_height = max_height
        display_width = max_height * aspect

    return Image(img_buffer, width=display_width, height=display_height)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Generador de PDF
# ──────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Generador de reportes PDF médicos profesionales.

    Uso:
        gen = ReportGenerator()
        pdf_bytes = gen.generate(
            report=clinical_report,
            frontal_image=annotated_frontal,
            profile_image=annotated_profile,
            patient_name="Juan Pérez",
        )
    """

    def __init__(self) -> None:
        self.styles = _get_styles()

    def generate(
        self,
        report: ClinicalReport,
        frontal_image: Optional[np.ndarray] = None,
        profile_image: Optional[np.ndarray] = None,
        patient_name: str = "Paciente",
        patient_id: str = "",
        doctor_name: str = "Profesional",
        clinic_name: str = "FacialMetrics Pro",
        notes: str = "",
    ) -> bytes:
        """
        Generar reporte PDF completo.

        Returns:
            Bytes del PDF generado.
        """
        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
        )

        elements: list = []

        # === PORTADA ===
        elements.extend(self._build_header(
            patient_name, patient_id, doctor_name, clinic_name
        ))

        # === IMÁGENES ANOTADAS ===
        if frontal_image is not None:
            elements.append(Paragraph("Análisis Frontal", self.styles["subtitle"]))
            elements.append(_cv2_to_reportlab_image(frontal_image))
            elements.append(Spacer(1, 5 * mm))

        if profile_image is not None:
            elements.append(Paragraph("Análisis de Perfil", self.styles["subtitle"]))
            elements.append(_cv2_to_reportlab_image(profile_image))
            elements.append(Spacer(1, 5 * mm))

        # === TABLA DE MEDICIONES ===
        elements.extend(self._build_measurements_table(report))

        # === ALERTAS CLÍNICAS ===
        elements.extend(self._build_alerts_section(report))

        # === NOTAS ===
        if notes:
            elements.append(Paragraph("Notas del Profesional", self.styles["subtitle"]))
            elements.append(Paragraph(notes, self.styles["body"]))
            elements.append(Spacer(1, 5 * mm))

        # === DISCLAIMER ===
        elements.extend(self._build_disclaimer())

        # Construir PDF
        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes

    def _build_header(
        self,
        patient_name: str,
        patient_id: str,
        doctor_name: str,
        clinic_name: str,
    ) -> list:
        """Construir encabezado del reporte."""
        elements = []

        elements.append(Paragraph(clinic_name, self.styles["title"]))
        elements.append(Paragraph(
            "Reporte de Análisis Facial Antropométrico",
            self.styles["heading"],
        ))
        elements.append(Spacer(1, 3 * mm))

        # Datos del paciente en tabla
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        data = [
            ["Paciente:", patient_name, "Fecha:", now],
            ["ID:", patient_id or "—", "Profesional:", doctor_name],
        ]

        info_table = Table(data, colWidths=[25 * mm, 55 * mm, 30 * mm, 55 * mm])
        info_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#333333")),
            ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#333333")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 5 * mm))

        # Línea separadora
        line_data = [["" * 100]]
        line_table = Table(line_data, colWidths=[170 * mm])
        line_table.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 1, colors.HexColor("#1565c0")),
        ]))
        elements.append(line_table)
        elements.append(Spacer(1, 5 * mm))

        return elements

    def _build_measurements_table(self, report: ClinicalReport) -> list:
        """Construir tabla de mediciones."""
        elements = []
        elements.append(Paragraph("Tabla de Mediciones", self.styles["subtitle"]))

        data = [["Parámetro", "Valor", "Ideal / Rango", "Estado"]]

        # Tercios
        if report.thirds:
            t = report.thirds
            data.append(["TERCIOS FACIALES", "", "", ""])
            data.append([
                "  Altura facial total",
                f"{t.total_height_mm:.1f} mm",
                "170-200 mm",
                "—",
            ])

            for seg in [t.upper, t.middle, t.lower]:
                status = "✓" if abs(seg.deviation) < 10 else "⚠" if abs(seg.deviation) < 15 else "✗"
                data.append([
                    f"  {seg.name}",
                    f"{seg.height_mm:.1f} mm ({seg.percentage:.1f}%)",
                    "33.3%",
                    status,
                ])

            data.append([
                "  Ratio labio/mentón",
                f"{t.lower_lip_ratio:.2f}",
                "0.50 (1:2)",
                "✓" if abs(t.lower_lip_ratio - 0.5) < 0.1 else "⚠",
            ])

        # Quintos
        if report.fifths:
            f = report.fifths
            data.append(["QUINTOS FACIALES", "", "", ""])
            data.append([
                "  Ancho facial total",
                f"{f.total_width_mm:.1f} mm",
                "130-150 mm",
                "—",
            ])

            for seg in f.segments:
                status = "✓" if abs(seg.deviation) < 8 else "⚠" if abs(seg.deviation) < 15 else "✗"
                data.append([
                    f"  {seg.name}",
                    f"{seg.width_mm:.1f} mm ({seg.percentage:.1f}%)",
                    "20%",
                    status,
                ])

        # Línea media
        if report.midline:
            m = report.midline
            data.append(["LÍNEA MEDIA", "", "", ""])
            data.append([
                "  Desviación nasal",
                f"{m.nasal_deviation_mm:.1f} mm ({m.nasal_deviation_side})",
                "< 2.0 mm",
                "✓" if m.nasal_deviation_mm < 2 else "⚠" if m.nasal_deviation_mm < 3 else "✗",
            ])
            data.append([
                "  Desviación mentón",
                f"{m.chin_deviation_mm:.1f} mm ({m.chin_deviation_side})",
                "< 3.0 mm",
                "✓" if m.chin_deviation_mm < 3 else "⚠" if m.chin_deviation_mm < 4 else "✗",
            ])

        # Perfil
        if report.profile:
            p = report.profile
            data.append(["ANÁLISIS DE PERFIL", "", "", ""])
            data.append([
                "  Proyección nasal",
                f"{p.nasal_projection_mm:.1f} mm",
                "15-20 mm",
                "—",
            ])
            data.append([
                "  Ángulo nasolabial",
                f"{p.nasolabial_angle:.1f}°",
                "90-110°",
                "✓" if 90 <= p.nasolabial_angle <= 110 else "⚠",
            ])
            data.append([
                "  Proyección mentón",
                f"{p.chin_projection_mm:.1f} mm",
                "-2 a +4 mm",
                "✓" if -2 <= p.chin_projection_mm <= 4 else "⚠",
            ])

        # Construir tabla
        col_widths = [55 * mm, 45 * mm, 35 * mm, 20 * mm]
        table = Table(data, colWidths=col_widths)

        # Estilo de la tabla
        style_commands = [
            # Header
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565c0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),

            # Body
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),

            # Grid
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("LINEBELOW", (0, 0), (-1, 0), 1.5, colors.HexColor("#0d47a1")),
        ]

        # Filas de sección (fondos)
        for i, row in enumerate(data):
            if row[0] in (
                "TERCIOS FACIALES", "QUINTOS FACIALES",
                "LÍNEA MEDIA", "ANÁLISIS DE PERFIL",
            ):
                style_commands.append(
                    ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e3f2fd"))
                )
                style_commands.append(
                    ("FONTNAME", (0, i), (0, i), "Helvetica-Bold")
                )
                style_commands.append(
                    ("SPAN", (0, i), (-1, i))
                )

            # Alternar fondos
            elif i > 0 and i % 2 == 0:
                style_commands.append(
                    ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fafafa"))
                )

        table.setStyle(TableStyle(style_commands))
        elements.append(table)
        elements.append(Spacer(1, 8 * mm))

        return elements

    def _build_alerts_section(self, report: ClinicalReport) -> list:
        """Construir sección de alertas clínicas."""
        elements = []

        if not report.alerts:
            elements.append(Paragraph("Alertas Clínicas", self.styles["subtitle"]))
            elements.append(Paragraph(
                "No se detectaron alertas clínicas significativas.",
                self.styles["alert_green"],
            ))
            return elements

        elements.append(Paragraph("Alertas Clínicas", self.styles["subtitle"]))

        for alert in report.alerts:
            severity_map = {
                AlertSeverity.ALERT: ("🔴", self.styles["alert_red"]),
                AlertSeverity.WARNING: ("🟡", self.styles["alert_yellow"]),
                AlertSeverity.INFO: ("🟢", self.styles["alert_green"]),
            }

            icon, style = severity_map[alert.severity]

            elements.append(Paragraph(
                f"<b>{icon} [{alert.code}]</b> {alert.message}",
                style,
            ))

            if alert.recommendation:
                elements.append(Paragraph(
                    f"    → Recomendación: {alert.recommendation}",
                    self.styles["small"],
                ))

            elements.append(Spacer(1, 2 * mm))

        elements.append(Spacer(1, 5 * mm))
        return elements

    def _build_disclaimer(self) -> list:
        """Construir sección de disclaimer legal."""
        elements = []

        # Línea separadora
        line_data = [[""]]
        line_table = Table(line_data, colWidths=[170 * mm])
        line_table.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ]))
        elements.append(line_table)
        elements.append(Spacer(1, 3 * mm))

        disclaimer_text = (
            "AVISO LEGAL: Este reporte ha sido generado de forma automatizada "
            "por el sistema FacialMetrics Pro y tiene carácter orientativo. "
            "Los valores, mediciones y sugerencias clínicas presentados NO "
            "constituyen un diagnóstico médico definitivo. Los resultados "
            "deben ser interpretados y validados por un profesional médico "
            "calificado. Las mediciones están sujetas a variaciones "
            "dependiendo de la calidad de la imagen, posición del paciente "
            "y calibración de la escala. El profesional tratante es el único "
            "responsable de las decisiones terapéuticas basadas en estos datos."
        )

        elements.append(Paragraph(disclaimer_text, self.styles["disclaimer"]))

        elements.append(Spacer(1, 3 * mm))
        elements.append(Paragraph(
            f"Generado por FacialMetrics Pro — {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            self.styles["small"],
        ))

        return elements


# ──────────────────────────────────────────────────────────────────────────────
# 4. Función de Conveniencia
# ──────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    report: ClinicalReport,
    frontal_image: Optional[np.ndarray] = None,
    profile_image: Optional[np.ndarray] = None,
    patient_name: str = "Paciente",
    patient_id: str = "",
    doctor_name: str = "Profesional",
    clinic_name: str = "FacialMetrics Pro",
    notes: str = "",
) -> bytes:
    """
    Función de conveniencia para generar el PDF en un solo paso.

    Returns:
        Bytes del PDF generado, listo para escritura o descarga.
    """
    generator = ReportGenerator()
    return generator.generate(
        report=report,
        frontal_image=frontal_image,
        profile_image=profile_image,
        patient_name=patient_name,
        patient_id=patient_id,
        doctor_name=doctor_name,
        clinic_name=clinic_name,
        notes=notes,
    )
