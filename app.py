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

import hashlib
import json
import os
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
    DEFAULT_MODEL,
    DENTAL_OPTIONS,
    PRESETS,
    PROCEDURES,
    FaceMesh3D,
    OpenRouterBackend,
    RefineError,
    active_changes,
    build_prompt,
    build_viewer_html,
    default_values,
    describe_for_ai,
    list_image_models,
    param_id,
    run_simulation,
    verify_geometry,
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


def image_digest(img: np.ndarray) -> str:
    """Huella de una imagen, para invalidar resultados cacheados en sesión."""
    return hashlib.md5(img.tobytes()).hexdigest()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_image_models(api_key: str) -> list[str]:
    return list_image_models(api_key)


def openrouter_default_key() -> str:
    try:
        return st.secrets.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
    except Exception:  # Sin secrets.toml
        return os.environ.get("OPENROUTER_API_KEY", "")


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
if "simulation_report" not in st.session_state:
    st.session_state.simulation_report = None
if "sim_ai" not in st.session_state:
    st.session_state.sim_ai = None
for _pid, _value in default_values().items():
    st.session_state.setdefault(f"sim_{_pid}", _value)


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
    st.markdown("### 🤖 IA generativa (OpenRouter)")
    openrouter_key = st.text_input(
        "API key",
        value=openrouter_default_key(),
        type="password",
        help="Se usa solo en esta sesión. También se lee de OPENROUTER_API_KEY "
             "(variable de entorno o .streamlit/secrets.toml).",
    )
    available_models = cached_image_models(openrouter_key) if openrouter_key else []
    model_options = sorted(set(available_models) | {DEFAULT_MODEL})
    openrouter_model = st.selectbox(
        "Modelo de imagen",
        model_options,
        index=model_options.index(DEFAULT_MODEL),
        help="Modelos de OpenRouter que generan imágenes a partir de imágenes.",
    )
    custom_model = st.text_input("Otro modelo (id)", placeholder="proveedor/modelo")
    if custom_model.strip():
        openrouter_model = custom_model.strip()

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

                # Controles de capas
                col_pc1, col_pc2, col_pc3, col_pc4, col_pc5 = st.columns(5)
                with col_pc1:
                    show_landmarks_p = st.checkbox("🔵 Puntos anatómicos", value=True, key="profile_show_landmarks")
                with col_pc2:
                    show_thirds_p = st.checkbox("➖ Tercios faciales", value=True, key="profile_show_thirds")
                with col_pc3:
                    show_fifths_p = st.checkbox("| Quintos faciales", value=True, key="profile_show_fifths")
                with col_pc4:
                    show_midline_p = st.checkbox("📐 Línea media", value=True, key="profile_show_midline")
                with col_pc5:
                    show_profile_p = st.checkbox("📏 Proyecciones", value=True, key="profile_show_projections")

                layers_p = {
                    "landmarks": show_landmarks_p,
                    "thirds": show_thirds_p,
                    "fifths": show_fifths_p,
                    "midline": show_midline_p,
                    "profile": show_profile_p,
                    "alerts": False,
                }

                # Tercios, quintos y línea media solo para dibujar: no se suman al
                # reporte de perfil (resultados, alertas y PDF no cambian)
                overlay_p = run_full_analysis(
                    st.session_state.landmarks_profile,
                    st.session_state.calibration,
                    view_type="frontal",
                )
                overlay_p.profile = report_p.profile

                annotated_p = render_full_analysis(
                    st.session_state.profile_img,
                    st.session_state.landmarks_profile,
                    overlay_p,
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

def apply_preset(values: dict[str, float]) -> None:
    for pid, value in default_values().items():
        st.session_state[f"sim_{pid}"] = values.get(pid, value)


def current_sim_values() -> dict[str, float]:
    return {pid: float(st.session_state[f"sim_{pid}"]) for pid in default_values()}


with tab_simulation:
    st.markdown("### Simulación Visual — Antes / Después")

    if st.session_state.calibration is None or st.session_state.landmarks_frontal is None:
        st.info("ℹ️ Cargá una imagen frontal, calibrá la escala y ejecutá el análisis primero.")
    else:
        calibration = st.session_state.calibration

        # ── Foto base ────────────────────────────────────────────────────
        base_options = ["Frontal en reposo"]
        if "smile_img" in st.session_state:
            base_options.append("Frontal en sonrisa")
        base_choice = st.radio("Foto base", base_options, horizontal=True)
        image_kind = "smile" if base_choice == "Frontal en sonrisa" else "rest"

        if image_kind == "smile":
            sim_image = st.session_state.smile_img
            digest = image_digest(sim_image)
            if st.session_state.get("landmarks_smile_digest") != digest:
                with st.spinner("Detectando landmarks en la sonrisa..."):
                    st.session_state.landmarks_smile = st.session_state.detector.detect(sim_image)
                    st.session_state.landmarks_smile_digest = digest
            sim_landmarks = st.session_state.landmarks_smile
        else:
            sim_image = st.session_state.frontal_img
            sim_landmarks = st.session_state.landmarks_frontal

        if sim_landmarks is None:
            st.error("❌ No se detectó un rostro en la foto seleccionada.")
        else:
            # ── Presets ──────────────────────────────────────────────────
            col_p1, col_p2, col_p3 = st.columns([3, 1, 1])
            with col_p1:
                preset = st.selectbox(
                    "Preset de tratamiento",
                    PRESETS,
                    format_func=lambda p: f"{p.name} — {p.description}",
                )
            with col_p2:
                st.button("Aplicar preset", on_click=apply_preset, args=(preset.values,),
                          use_container_width=True)
            with col_p3:
                st.button("Reiniciar", on_click=apply_preset, args=({},),
                          use_container_width=True)

            # ── Parámetros en mm / grados ────────────────────────────────
            col_controls, col_view = st.columns([1, 2])

            with col_controls:
                st.caption(f"Escala: {calibration.pixel_per_mm:.2f} px/mm ({calibration.method})")
                for proc in PROCEDURES:
                    if proc.image == "smile" and image_kind != "smile":
                        continue
                    active = any(
                        st.session_state[f"sim_{param_id(proc, par)}"] != 0 for par in proc.parameters
                    )
                    with st.expander(proc.name, expanded=active):
                        if proc.note:
                            st.caption(proc.note)
                        for par in proc.parameters:
                            st.slider(
                                f"{par.label} ({par.unit})",
                                min_value=par.min_value,
                                max_value=par.max_value,
                                step=par.step,
                                key=f"sim_{param_id(proc, par)}",
                                help=par.help or None,
                            )

                dental_choice: list[str] = []
                if image_kind == "smile":
                    with st.expander("Dental (solo con IA)"):
                        dental_choice = st.multiselect(
                            "Cambios dentales",
                            [opt.key for opt in DENTAL_OPTIONS],
                            format_func=lambda k: next(o.label for o in DENTAL_OPTIONS if o.key == k),
                            help="No son geométricos: se aplican en el refinado con IA.",
                        )

            values = current_sim_values()
            outcome = run_simulation(sim_image, sim_landmarks, calibration, values, image_kind)
            changes = active_changes(values)

            with col_view:
                col_before, col_after = st.columns(2)
                with col_before:
                    st.image(img_to_rgb(sim_image), caption="Antes", use_container_width=True)
                with col_after:
                    st.image(img_to_rgb(outcome.image), caption="Después (simulación geométrica)",
                             use_container_width=True)

                if outcome.applied_fraction < 0.999:
                    st.warning(
                        f"⚠️ Los valores superan lo que la foto admite sin plegar la imagen: "
                        f"se aplicó el {outcome.applied_fraction:.0%} del cambio. "
                        "Una foto de mayor resolución permite cambios mayores."
                    )

                changed = [d for d in outcome.deltas if abs(d.delta) >= 0.1]
                if changed:
                    st.markdown("##### 📏 Re-medición")
                    st.dataframe(
                        {
                            "Medida": [d.name for d in changed],
                            "Antes (mm)": [f"{d.before:.1f}" for d in changed],
                            "Después (mm)": [f"{d.after:.1f}" for d in changed],
                            "Δ (mm)": [f"{d.delta:+.1f}" for d in changed],
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Las proyecciones de perfil se estiman con la profundidad de la malla 3D: "
                        "el valor absoluto es aproximado, la diferencia refleja el cambio simulado."
                    )

            # ── Visor 3D ─────────────────────────────────────────────────
            st.divider()
            st.markdown("#### 🧊 Visor 3D")
            st.caption(
                "Girá el rostro con el mouse (hasta ±40°). Los cambios de proyección "
                "(mentón, dorso, pómulos, ojeras) se aprecian al girarlo."
            )
            depth_scale = st.slider("Profundidad de la malla", 0.5, 2.0, 1.0, 0.1)
            st.iframe(
                build_viewer_html(outcome.mesh_before, outcome.mesh_after, sim_image,
                                  height=520, depth_scale=depth_scale),
                height=530,
            )

            # ── Refinado con IA ──────────────────────────────────────────
            st.divider()
            st.markdown("#### ✨ Refinado fotorrealista con IA")
            ai_changes = describe_for_ai(values, dental_choice)
            request_key = json.dumps(
                [image_kind, image_digest(sim_image), values, dental_choice, openrouter_model],
                sort_keys=True,
            )

            consent = st.checkbox(
                "El paciente firmó el consentimiento para procesar su foto con un servicio "
                "externo de IA (OpenRouter y el proveedor del modelo).",
                key="sim_ai_consent",
            )
            with st.expander("Ver prompt"):
                st.code(build_prompt(ai_changes), language="text")

            can_refine = bool(openrouter_key) and consent and bool(ai_changes)
            if not openrouter_key:
                st.caption("Ingresá la API key de OpenRouter en la barra lateral.")
            elif not ai_changes:
                st.caption("Configurá al menos un cambio para refinar.")

            if st.button("✨ Refinar con IA", type="primary", disabled=not can_refine):
                with st.spinner(f"Generando con {openrouter_model}..."):
                    try:
                        backend = OpenRouterBackend(api_key=openrouter_key, model=openrouter_model)
                        ai_image = backend.refine(sim_image, outcome.image, build_prompt(ai_changes))
                        detected = st.session_state.detector.detect(ai_image)
                        verification = verify_geometry(
                            FaceMesh3D.from_landmarks(detected) if detected else None,
                            outcome.mesh_after,
                            calibration.pixel_per_mm,
                        )
                        st.session_state.sim_ai = {
                            "key": request_key,
                            "image": ai_image,
                            "model": openrouter_model,
                            "verification": verification,
                        }
                    except RefineError as e:
                        st.error(f"❌ {e}")

            sim_ai = st.session_state.sim_ai
            ai_current = sim_ai is not None and sim_ai["key"] == request_key
            if sim_ai is not None and not ai_current:
                st.info("ℹ️ Cambiaste la simulación: el resultado de IA anterior ya no corresponde.")

            if ai_current:
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.image(img_to_rgb(sim_image), caption="Antes", use_container_width=True)
                with col_a2:
                    st.image(img_to_rgb(outcome.image), caption="Simulación geométrica",
                             use_container_width=True)
                with col_a3:
                    st.image(img_to_rgb(sim_ai["image"]), caption=f"IA — {sim_ai['model']}",
                             use_container_width=True)

                v = sim_ai["verification"]
                if not v.face_detected:
                    st.error("❌ No se detectó el rostro en la imagen generada. No es confiable.")
                elif v.passed:
                    st.success(
                        f"✅ La imagen de IA respeta la geometría simulada "
                        f"(error medio {v.mean_error_mm:.2f} mm, máx. {v.max_error_mm:.1f} mm)."
                    )
                else:
                    st.warning(
                        f"⚠️ La IA se apartó de la geometría simulada (error medio "
                        f"{v.mean_error_mm:.2f} mm, máx. {v.max_error_mm:.1f} mm). "
                        "Reintentá o usá la simulación geométrica."
                    )

            # ── Resultado para el reporte ────────────────────────────────
            if changes or dental_choice:
                final_options = ["Simulación geométrica"]
                if ai_current:
                    final_options.append("Refinado con IA")
                final_choice = st.radio("Imagen final para el reporte", final_options, horizontal=True)
                use_ai = final_choice == "Refinado con IA"

                st.session_state.simulation_report = {
                    "before": sim_image,
                    "after": sim_ai["image"] if use_ai else outcome.image,
                    "source": f"IA ({sim_ai['model']})" if use_ai else "Simulación geométrica",
                    "changes": [
                        (proc.name, par.label, value, par.unit) for proc, par, value in changes
                    ] + [
                        ("Dental (IA)", opt.label, None, "")
                        for opt in DENTAL_OPTIONS if use_ai and opt.key in dental_choice
                    ],
                    "deltas": [
                        (d.name, d.before, d.after) for d in outcome.deltas if abs(d.delta) >= 0.1
                    ],
                }
            else:
                st.session_state.simulation_report = None


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
                    simulation=st.session_state.simulation_report,
                )

            st.success("✅ Reporte PDF generado exitosamente.")

            st.download_button(
                "⬇️ Descargar Reporte PDF",
                data=pdf_bytes,
                file_name=f"reporte_facial_{patient_name or 'paciente'}.pdf",
                mime="application/pdf",
                type="primary",
            )
