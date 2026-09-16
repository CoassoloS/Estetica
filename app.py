"""
app.py — Aplicación Web Interactiva de Análisis Facial Antropométrico.

Aplicación Streamlit con las siguientes secciones:
    1. Carga de Imágenes (frontal reposo, sonrisa, perfil)
    2. Calibración de escala (manual / automática)
    3. Análisis Facial con overlays interactivos
    4. Resultados (tabla + alertas + JSON)
    5. Simulación antes/después
    6. Generación y descarga de PDF

Uso:
    streamlit run app.py

Autor: FacialMetrics Pro
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from landmarks import (
    AnthropometricPoint,
    FaceLandmarkDetector,
    LandmarkResult,
    get_display_name,
)
from measurements import (
    ClinicalReport,
    FacialAnalyzer,
    ClinicalRuleEngine,
    ScaleCalibrator,
    CalibrationResult,
    AlertSeverity,
    run_full_analysis,
)
from drawer import (
    DrawConfig,
    draw_landmarks,
    draw_thirds,
    draw_fifths,
    draw_midline,
    draw_profile,
    draw_alerts_panel,
    render_full_analysis,
    render_comparison,
)
from report import generate_pdf_report
from simulation import (
    MeshWarper,
    SimulationConfig,
    generate_nose_warp_points,
    generate_chin_warp_points,
    generate_lips_warp_points,
    create_region_mask,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuración de Página
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FacialMetrics Pro — Análisis Facial",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1565c0 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(21, 101, 192, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #252540 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.3rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #90caf9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e3f2fd;
    }
    .metric-card .sub {
        font-size: 0.7rem;
        color: #78909c;
        margin-top: 0.2rem;
    }

    /* Alert badges */
    .alert-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .alert-red { background: rgba(198, 40, 40, 0.15); color: #ef5350; border: 1px solid rgba(198,40,40,0.3); }
    .alert-yellow { background: rgba(230, 81, 0, 0.15); color: #ff9800; border: 1px solid rgba(230,81,0,0.3); }
    .alert-green { background: rgba(46, 125, 50, 0.15); color: #66bb6a; border: 1px solid rgba(46,125,50,0.3); }

    /* Tabs refinement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
    }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9;
    }

    /* Image container */
    .image-container {
        border: 2px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        font-size: 0.75rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔬 FacialMetrics Pro</h1>
    <p>Sistema de Análisis Facial Cefalométrico / Antropométrico — Prototipo Clínico</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def load_image(uploaded_file) -> Optional[np.ndarray]:
    """Cargar imagen desde Streamlit UploadedFile."""
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)  # Reset para posibles re-lecturas
    return img


def img_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convertir BGR a RGB para Streamlit."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_metric_card(label: str, value: str, sub: str = "") -> str:
    """Crear HTML de tarjeta métrica."""
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """


def severity_badge(severity: AlertSeverity, text: str) -> str:
    """Crear badge HTML de severidad."""
    css_class = {
        AlertSeverity.ALERT: "alert-red",
        AlertSeverity.WARNING: "alert-yellow",
        AlertSeverity.INFO: "alert-green",
    }[severity]
    icon = {
        AlertSeverity.ALERT: "🔴",
        AlertSeverity.WARNING: "🟡",
        AlertSeverity.INFO: "🟢",
    }[severity]
    return f'<span class="alert-badge {css_class}">{icon} {text}</span>'


# ──────────────────────────────────────────────────────────────────────────────
# Estado de Sesión
# ──────────────────────────────────────────────────────────────────────────────

if "detector" not in st.session_state:
    st.session_state.detector = FaceLandmarkDetector()
if "calibration" not in st.session_state:
    st.session_state.calibration = None
if "landmarks_frontal" not in st.session_state:
    st.session_state.landmarks_frontal = None
if "landmarks_profile" not in st.session_state:
    st.session_state.landmarks_profile = None
if "report_frontal" not in st.session_state:
    st.session_state.report_frontal = None
if "report_profile" not in st.session_state:
    st.session_state.report_profile = None


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Datos del Paciente
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 👤 Datos del Paciente")
    patient_name = st.text_input("Nombre", value="", placeholder="Nombre del paciente")
    patient_id = st.text_input("ID / Historia Clínica", value="", placeholder="HC-0001")
    doctor_name = st.text_input("Profesional", value="", placeholder="Dr./Dra.")
    clinic_name = st.text_input("Clínica", value="FacialMetrics Pro")

    st.divider()
    st.markdown("### ⚙️ Configuración")
    trichion_ratio = st.slider(
        "Ratio estimación Trichion",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.01,
        help="Proporción de la distancia G-Me para estimar Trichion por encima de G"
    )

    intercantal_ref = st.number_input(
        "Distancia intercantal ref. (mm)",
        min_value=20.0,
        max_value=45.0,
        value=32.0,
        step=0.5,
        help="Distancia intercantal de referencia para calibración automática"
    )

    st.divider()
    st.markdown(
        '<div class="footer">FacialMetrics Pro v0.1<br>'
        'Prototipo de análisis — No sustituye diagnóstico médico</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tabs Principales
# ──────────────────────────────────────────────────────────────────────────────

tab_upload, tab_calibrate, tab_analysis, tab_results, tab_simulation, tab_report = st.tabs([
    "📸 Carga de Imágenes",
    "📐 Calibración",
    "🔬 Análisis Facial",
    "📊 Resultados",
    "🔄 Simulación",
    "📄 Reporte PDF",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Carga de Imágenes
# ══════════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.markdown("### Cargar Imágenes Clínicas")
    st.info(
        "📋 **Requisitos de imagen**: Fotos estandarizadas con fondo neutro, "
        "iluminación uniforme, paciente en posición natural. "
        "Se recomienda incluir una regla visible para calibración precisa."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Frontal en Reposo")
        frontal_file = st.file_uploader(
            "Subir frontal reposo",
            type=["jpg", "jpeg", "png", "bmp"],
            key="frontal_upload",
            label_visibility="collapsed",
        )
        if frontal_file:
            frontal_img = load_image(frontal_file)
            if frontal_img is not None:
                st.session_state.frontal_img = frontal_img
                st.image(img_to_rgb(frontal_img), caption="Frontal Reposo", use_container_width=True)
                st.success(f"✅ {frontal_img.shape[1]}×{frontal_img.shape[0]} px")

    with col2:
        st.markdown("#### Frontal en Sonrisa")
        smile_file = st.file_uploader(
            "Subir frontal sonrisa",
            type=["jpg", "jpeg", "png", "bmp"],
            key="smile_upload",
            label_visibility="collapsed",
        )
        if smile_file:
            smile_img = load_image(smile_file)
            if smile_img is not None:
                st.session_state.smile_img = smile_img
                st.image(img_to_rgb(smile_img), caption="Frontal Sonrisa", use_container_width=True)
                st.success(f"✅ {smile_img.shape[1]}×{smile_img.shape[0]} px")

    with col3:
        st.markdown("#### Perfil Lateral")
        profile_file = st.file_uploader(
            "Subir perfil lateral",
            type=["jpg", "jpeg", "png", "bmp"],
            key="profile_upload",
            label_visibility="collapsed",
        )
        if profile_file:
            profile_img = load_image(profile_file)
            if profile_img is not None:
                st.session_state.profile_img = profile_img
                st.image(img_to_rgb(profile_img), caption="Perfil Lateral", use_container_width=True)
                st.success(f"✅ {profile_img.shape[1]}×{profile_img.shape[0]} px")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Calibración
# ══════════════════════════════════════════════════════════════════════════════

with tab_calibrate:
    st.markdown("### Calibración de Escala (Píxeles → Milímetros)")

    cal_method = st.radio(
        "Método de calibración",
        ["🔲 Automático (Distancia Intercantal)", "📏 Manual (Regla en foto)"],
        horizontal=True,
    )

    if cal_method.startswith("🔲"):
        st.markdown(
            f"Usando distancia intercantal de referencia: **{intercantal_ref} mm**"
        )
        st.caption(
            "La calibración automática se ejecutará al detectar landmarks en la "
            "imagen frontal. Usa la distancia entre los cantos internos de los "
            "ojos como referencia de escala."
        )

        if st.button("⚡ Calibrar Automáticamente", type="primary"):
            if "frontal_img" not in st.session_state:
                st.error("⚠️ Primero cargá una imagen frontal en la pestaña anterior.")
            else:
                with st.spinner("Detectando landmarks y calibrando..."):
                    detector = st.session_state.detector
                    detector._trichion_ratio = trichion_ratio
                    result = detector.detect(st.session_state.frontal_img)

                    if result is None:
                        st.error("❌ No se detectó ningún rostro en la imagen.")
                    else:
                        st.session_state.landmarks_frontal = result
                        cal = ScaleCalibrator.calibrate_intercantal(
                            result, intercantal_ref
                        )
                        st.session_state.calibration = cal

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(create_metric_card(
                                "Factor de escala",
                                f"{cal.mm_per_pixel:.4f} mm/px",
                                f"Método: {cal.method}",
                            ), unsafe_allow_html=True)
                        with col_b:
                            st.markdown(create_metric_card(
                                "Dist. Intercantal",
                                f"{cal.reference_distance_px:.1f} px",
                                f"= {cal.reference_distance_mm:.1f} mm ref.",
                            ), unsafe_allow_html=True)
                        with col_c:
                            st.markdown(create_metric_card(
                                "Resolución efectiva",
                                f"{1/cal.mm_per_pixel:.1f} px/mm",
                                "Precisión de la imagen",
                            ), unsafe_allow_html=True)

                        st.success("✅ Calibración completada exitosamente.")

    else:  # Manual
        st.markdown("Ingresá la distancia real entre dos puntos conocidos de la foto:")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            p1_x = st.number_input("Punto 1 — X (px)", value=0, min_value=0)
            p1_y = st.number_input("Punto 1 — Y (px)", value=0, min_value=0)
        with col_m2:
            p2_x = st.number_input("Punto 2 — X (px)", value=100, min_value=0)
            p2_y = st.number_input("Punto 2 — Y (px)", value=0, min_value=0)

        real_dist = st.number_input(
            "Distancia real entre los puntos (mm)",
            value=30.0,
            min_value=1.0,
            step=0.5,
        )

        if st.button("📏 Calibrar Manualmente", type="primary"):
            try:
                cal = ScaleCalibrator.calibrate_manual(
                    (p1_x, p1_y), (p2_x, p2_y), real_dist
                )
                st.session_state.calibration = cal
                st.success(
                    f"✅ Calibración manual: {cal.mm_per_pixel:.4f} mm/px "
                    f"({1/cal.mm_per_pixel:.1f} px/mm)"
                )
            except ValueError as e:
                st.error(f"❌ Error de calibración: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Análisis Facial
# ══════════════════════════════════════════════════════════════════════════════

with tab_analysis:
    st.markdown("### Análisis Facial Interactivo")

    if st.session_state.calibration is None:
        st.warning("⚠️ Primero calibrá la escala en la pestaña 📐 Calibración.")
    elif "frontal_img" not in st.session_state:
        st.warning("⚠️ Primero cargá una imagen frontal en la pestaña 📸 Carga de Imágenes.")
    else:
        # Detectar landmarks si no se hizo
        if st.session_state.landmarks_frontal is None:
            with st.spinner("Detectando landmarks faciales..."):
                detector = st.session_state.detector
                detector._trichion_ratio = trichion_ratio
                result = detector.detect(st.session_state.frontal_img)
                if result:
                    st.session_state.landmarks_frontal = result

        # Análisis frontal
        if st.session_state.landmarks_frontal:
            st.markdown("#### Vista Frontal")

            # Controles de capas
            col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
            with col_ctrl1:
                show_landmarks = st.checkbox("🔵 Puntos anatómicos", value=True)
            with col_ctrl2:
                show_thirds = st.checkbox("➖ Tercios faciales", value=True)
            with col_ctrl3:
                show_fifths = st.checkbox("| Quintos faciales", value=True)
            with col_ctrl4:
                show_midline = st.checkbox("📐 Línea media", value=True)

            # Ejecutar análisis
            report = run_full_analysis(
                st.session_state.landmarks_frontal,
                st.session_state.calibration,
                view_type="frontal",
            )
            st.session_state.report_frontal = report

            # Renderizar overlays
            layers = {
                "landmarks": show_landmarks,
                "thirds": show_thirds,
                "fifths": show_fifths,
                "midline": show_midline,
                "profile": False,
                "alerts": False,
            }

            annotated = render_full_analysis(
                st.session_state.frontal_img,
                st.session_state.landmarks_frontal,
                report,
                layers=layers,
            )

            # Mostrar imagen
            col_img1, col_img2 = st.columns([2, 1])
            with col_img1:
                st.image(
                    img_to_rgb(annotated),
                    caption="Análisis Frontal",
                    use_container_width=True,
                )

            with col_img2:
                # Métricas rápidas
                if report.thirds:
                    st.markdown("##### Tercios Faciales")
                    t = report.thirds
                    st.markdown(create_metric_card(
                        "Altura Facial Total",
                        f"{t.total_height_mm:.1f} mm",
                        f"Tr→Me",
                    ), unsafe_allow_html=True)

                    for seg in [t.upper, t.middle, t.lower]:
                        icon = "✅" if abs(seg.deviation) < 10 else "⚠️" if abs(seg.deviation) < 15 else "❌"
                        st.markdown(create_metric_card(
                            seg.name,
                            f"{seg.height_mm:.1f} mm ({seg.percentage:.1f}%)",
                            f"{icon} Ideal: 33.3% | Desv: {seg.deviation:+.1f}%",
                        ), unsafe_allow_html=True)

                if report.midline:
                    st.markdown("##### Línea Media")
                    m = report.midline
                    st.markdown(create_metric_card(
                        "Desviación Nasal",
                        f"{m.nasal_deviation_mm:.1f} mm",
                        f"Hacia: {m.nasal_deviation_side}",
                    ), unsafe_allow_html=True)
                    st.markdown(create_metric_card(
                        "Desviación Mentón",
                        f"{m.chin_deviation_mm:.1f} mm",
                        f"Hacia: {m.chin_deviation_side}",
                    ), unsafe_allow_html=True)

        # Análisis de perfil
        if "profile_img" in st.session_state:
            st.divider()
            st.markdown("#### Vista de Perfil Lateral")

            if st.session_state.landmarks_profile is None:
                with st.spinner("Detectando landmarks en perfil..."):
                    detector = st.session_state.detector
                    result_p = detector.detect(st.session_state.profile_img)
                    if result_p:
                        st.session_state.landmarks_profile = result_p

            if st.session_state.landmarks_profile:
                report_p = run_full_analysis(
                    st.session_state.landmarks_profile,
                    st.session_state.calibration,
                    view_type="profile",
                )
                st.session_state.report_profile = report_p

                layers_p = {
                    "landmarks": True,
                    "thirds": False,
                    "fifths": False,
                    "midline": False,
                    "profile": True,
                    "alerts": False,
                }

                annotated_p = render_full_analysis(
                    st.session_state.profile_img,
                    st.session_state.landmarks_profile,
                    report_p,
                    layers=layers_p,
                )

                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    st.image(
                        img_to_rgb(annotated_p),
                        caption="Análisis de Perfil",
                        use_container_width=True,
                    )

                with col_p2:
                    if report_p.profile:
                        p = report_p.profile
                        st.markdown("##### Proyecciones Anteroposteriores")
                        st.markdown(create_metric_card(
                            "Proyección Nasal",
                            f"{p.nasal_projection_mm:.1f} mm",
                            "Ref: vertical Sn",
                        ), unsafe_allow_html=True)
                        st.markdown(create_metric_card(
                            "Ángulo Nasolabial",
                            f"{p.nasolabial_angle:.1f}°",
                            "Normal: 90-110°",
                        ), unsafe_allow_html=True)
                        st.markdown(create_metric_card(
                            "Proyección Mentón",
                            f"{p.chin_projection_mm:.1f} mm",
                            "Normal: -2 a +4 mm",
                        ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Resultados
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    st.markdown("### Resultados del Análisis")

    report_f = st.session_state.report_frontal
    report_p = st.session_state.report_profile

    if report_f is None and report_p is None:
        st.info("ℹ️ Ejecutá el análisis en la pestaña 🔬 para ver los resultados.")
    else:
        # Alertas clínicas
        all_alerts = []
        if report_f and report_f.alerts:
            all_alerts.extend(report_f.alerts)
        if report_p and report_p.alerts:
            all_alerts.extend(report_p.alerts)

        if all_alerts:
            st.markdown("#### 🚨 Alertas Clínicas")
            for alert in all_alerts:
                badge = severity_badge(alert.severity, alert.code)
                st.markdown(
                    f"{badge} **{alert.message}**",
                    unsafe_allow_html=True,
                )
                if alert.recommendation:
                    st.caption(f"  → {alert.recommendation}")
        else:
            st.success("✅ No se detectaron alertas clínicas significativas.")

        st.divider()

        # Tabla de mediciones
        st.markdown("#### 📏 Tabla de Mediciones")

        if report_f and report_f.thirds:
            t = report_f.thirds
            thirds_data = {
                "Segmento": [t.upper.name, t.middle.name, t.lower.name],
                "Altura (mm)": [
                    f"{t.upper.height_mm:.1f}",
                    f"{t.middle.height_mm:.1f}",
                    f"{t.lower.height_mm:.1f}",
                ],
                "Porcentaje": [
                    f"{t.upper.percentage:.1f}%",
                    f"{t.middle.percentage:.1f}%",
                    f"{t.lower.percentage:.1f}%",
                ],
                "Ideal": ["33.3%"] * 3,
                "Desviación": [
                    f"{t.upper.deviation:+.1f}%",
                    f"{t.middle.deviation:+.1f}%",
                    f"{t.lower.deviation:+.1f}%",
                ],
            }
            st.markdown("**Tercios Faciales**")
            st.dataframe(thirds_data, use_container_width=True, hide_index=True)

        if report_f and report_f.fifths:
            f = report_f.fifths
            fifths_data = {
                "Segmento": [s.name for s in f.segments],
                "Ancho (mm)": [f"{s.width_mm:.1f}" for s in f.segments],
                "Porcentaje": [f"{s.percentage:.1f}%" for s in f.segments],
                "Ideal": ["20.0%"] * 5,
                "Desviación": [f"{s.deviation:+.1f}%" for s in f.segments],
            }
            st.markdown("**Quintos Faciales**")
            st.dataframe(fifths_data, use_container_width=True, hide_index=True)

        st.divider()

        # JSON exportable
        st.markdown("#### 📋 JSON de Métricas")
        combined_report = {}
        if report_f:
            combined_report["frontal"] = report_f.to_dict()
        if report_p:
            combined_report["profile"] = report_p.to_dict()

        json_str = json.dumps(combined_report, indent=2, ensure_ascii=False)
        st.json(combined_report)

        st.download_button(
            "⬇️ Descargar JSON",
            data=json_str,
            file_name=f"facialmetrics_{patient_name or 'paciente'}.json",
            mime="application/json",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Simulación
# ══════════════════════════════════════════════════════════════════════════════

with tab_simulation:
    st.markdown("### Simulación Visual — Antes / Después")

    if "frontal_img" not in st.session_state or st.session_state.landmarks_frontal is None:
        st.info(
            "ℹ️ Cargá una imagen frontal y ejecutá el análisis primero."
        )
    else:
        landmarks = st.session_state.landmarks_frontal
        original = st.session_state.frontal_img

        st.markdown("#### Controles de Simulación")

        sim_region = st.selectbox(
            "Zona a modificar",
            ["Nariz (Rinoplastia)", "Mentón (Mentoplastia)", "Labios (Queiloplastia)"],
        )

        col_s1, col_s2 = st.columns(2)

        warp_points = []

        if "Nariz" in sim_region:
            with col_s1:
                nose_dx = st.slider("Nariz — Desplazamiento Horizontal (px)", -30, 30, 0)
                nose_dy = st.slider("Nariz — Desplazamiento Vertical (px)", -20, 20, 0)
            with col_s2:
                nose_scale = st.slider("Nariz — Escala", 0.7, 1.3, 1.0, 0.05)

            warp_points = generate_nose_warp_points(landmarks, nose_dx, nose_dy, nose_scale)

        elif "Mentón" in sim_region:
            with col_s1:
                chin_dx = st.slider("Mentón — Avance/Retroceso (px)", -30, 30, 0)
            with col_s2:
                chin_dy = st.slider("Mentón — Alargamiento/Acortamiento (px)", -20, 20, 0)

            warp_points = generate_chin_warp_points(landmarks, chin_dx, chin_dy)

        elif "Labios" in sim_region:
            with col_s1:
                lip_upper = st.slider("Labio Superior — Volumen (px)", -15, 15, 0)
                lip_lower = st.slider("Labio Inferior — Volumen (px)", -15, 15, 0)
            with col_s2:
                lip_width = st.slider("Comisuras — Escala", 0.85, 1.15, 1.0, 0.01)

            warp_points = generate_lips_warp_points(
                landmarks, -lip_upper, lip_lower, lip_width
            )

        # Aplicar warp
        if warp_points and any(
            wp.original != wp.displaced for wp in warp_points
        ):
            with st.spinner("Aplicando deformación..."):
                warper = MeshWarper(SimulationConfig(smoothness=30.0))
                simulated = warper.warp(original, warp_points)

            # Side-by-side
            comparison = render_comparison(original, simulated, "Antes", "Después")
            st.image(
                img_to_rgb(comparison),
                caption="Comparación Antes / Después",
                use_container_width=True,
            )

            st.session_state.simulated_img = simulated
        else:
            st.image(
                img_to_rgb(original),
                caption="Imagen original — Mové los sliders para simular",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: Reporte PDF
# ══════════════════════════════════════════════════════════════════════════════

with tab_report:
    st.markdown("### Generación de Reporte PDF")

    notes = st.text_area(
        "Notas del profesional (opcional)",
        placeholder="Observaciones clínicas adicionales...",
        height=100,
    )

    if st.button("📄 Generar Reporte PDF", type="primary"):
        report_f = st.session_state.report_frontal
        report_p = st.session_state.report_profile

        if report_f is None and report_p is None:
            st.error("⚠️ Ejecutá el análisis primero en la pestaña 🔬")
        else:
            with st.spinner("Generando PDF profesional..."):
                # Combinar reportes
                combined = ClinicalReport(
                    calibration=st.session_state.calibration,
                )
                if report_f:
                    combined.thirds = report_f.thirds
                    combined.fifths = report_f.fifths
                    combined.midline = report_f.midline
                    combined.alerts = report_f.alerts.copy()
                if report_p:
                    combined.profile = report_p.profile
                    if report_p.alerts:
                        combined.alerts.extend(report_p.alerts)

                # Generar imágenes anotadas para el PDF
                annotated_frontal = None
                annotated_profile = None

                if "frontal_img" in st.session_state and st.session_state.landmarks_frontal:
                    annotated_frontal = render_full_analysis(
                        st.session_state.frontal_img,
                        st.session_state.landmarks_frontal,
                        combined,
                        layers={"landmarks": True, "thirds": True, "fifths": True, "midline": True, "profile": False, "alerts": False},
                    )

                if "profile_img" in st.session_state and st.session_state.landmarks_profile:
                    annotated_profile = render_full_analysis(
                        st.session_state.profile_img,
                        st.session_state.landmarks_profile,
                        combined,
                        layers={"landmarks": True, "thirds": False, "fifths": False, "midline": False, "profile": True, "alerts": False},
                    )

                pdf_bytes = generate_pdf_report(
                    report=combined,
                    frontal_image=annotated_frontal,
                    profile_image=annotated_profile,
                    patient_name=patient_name or "Paciente",
                    patient_id=patient_id,
                    doctor_name=doctor_name or "Profesional",
                    clinic_name=clinic_name,
                    notes=notes,
                )

            st.success("✅ Reporte PDF generado exitosamente.")

            st.download_button(
                "⬇️ Descargar Reporte PDF",
                data=pdf_bytes,
                file_name=f"reporte_facial_{patient_name or 'paciente'}.pdf",
                mime="application/pdf",
                type="primary",
            )
