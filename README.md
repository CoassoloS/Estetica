# FacialMetrics Pro 🔬

Sistema de análisis facial cefalométrico/antropométrico automatizado para clínicas de estética facial y odontológica.

## Descripción

El sistema procesa fotos médicas estandarizadas (frontal en reposo, frontal en sonrisa y perfil estricto) para:

- **Detectar puntos anatómicos** (landmarks) con MediaPipe Face Mesh (478 puntos)
- **Calibrar escala** píxeles → milímetros (manual con regla, o automático por distancia intercantal)
- **Calcular proporciones faciales**: Ley de Tercios, Ley de Quintos, Línea Media, Análisis de Perfil
- **Emitir diagnósticos preliminares** con motor de reglas clínicas (alertas de asimetría, sonrisa gingival, etc.)
- **Simular resultados estéticos** con deformación Thin Plate Splines (rinoplastia, mentoplastia, queiloplastia)
- **Generar reportes PDF** profesionales con imágenes anotadas, tablas de mediciones y alertas clínicas

## Stack Tecnológico

| Componente | Tecnología |
|:---|:---|
| Procesamiento de imagen | OpenCV, MediaPipe, NumPy, Pillow |
| Cálculos científicos | SciPy (RBF/TPS) |
| Frontend | Streamlit |
| Reportes PDF | ReportLab |
| Lenguaje | Python 3.10+ |

## Estructura del Proyecto

```
Estetica/
├── app.py                 # Aplicación Streamlit principal
├── landmarks.py           # Mapeo MediaPipe → Puntos Antropométricos
├── measurements.py        # Calibración + Cálculos + Motor de Reglas
├── drawer.py              # Renderizado de overlays clínicos
├── report.py              # Generación de PDF con ReportLab
├── simulation.py          # Simulación visual (TPS warp + stub Inpainting)
├── requirements.txt       # Dependencias
├── README.md              # Este archivo
└── tests/
    ├── test_landmarks.py
    └── test_measurements.py
```

## Instalación y Ejecución

### 1. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate     # Linux/macOS
# venv\Scripts\activate      # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

### 4. Ejecutar tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Uso

### Flujo de Trabajo

1. **📸 Carga de Imágenes**: Subir foto frontal en reposo, frontal en sonrisa (opcional) y perfil lateral (opcional).
2. **📐 Calibración**: Calibrar escala automáticamente (distancia intercantal ~32mm) o manualmente (con regla visible en la foto).
3. **🔬 Análisis Facial**: Visualizar landmarks detectados y activar/desactivar capas de overlay (tercios, quintos, línea media).
4. **📊 Resultados**: Ver tabla de mediciones, alertas clínicas con semáforo (🟢🟡🔴) y exportar JSON de métricas.
5. **🔄 Simulación**: Simular cambios estéticos (rinoplastia, mentoplastia, queiloplastia) con sliders interactivos.
6. **📄 Reporte**: Generar y descargar PDF profesional con imágenes anotadas, mediciones y alertas.

### Puntos Anatómicos Detectados

| Punto | Abreviatura | Descripción |
|:---|:---:|:---|
| Trichion | Tr | Nacimiento del pelo (estimado) |
| Glabella | G | Punto más prominente entre cejas |
| Nasion | N | Raíz nasal |
| Pronasale | Pn | Punta nasal |
| Subnasale | Sn | Base nasal |
| Stomion Superior | Sts | Centro labio superior |
| Stomion Inferior | Sti | Centro labio inferior |
| Pogonion | Pog' | Mentón anterior |
| Menton | Me | Mentón inferior |
| Endocanthion L/R | En | Cantos internos de los ojos |
| Exocanthion L/R | Ex | Cantos externos de los ojos |

## Alertas Clínicas

El motor de reglas evalúa automáticamente:

| Condición | Umbral | Alerta |
|:---|:---:|:---|
| Desviación nasal | > 2 mm | Asimetría nasal / Evaluar Rinoplastia |
| Exposición gingival | > 3 mm | Sonrisa gingival |
| Desbalance de tercios | > 15% | Discrepancia en tercio / Evaluar mentoplastia |
| Desviación del mentón | > 3 mm | Evaluar mentoplastia |
| Ratio labio/mentón | > ±20% de 1:2 | Evaluar queiloplastia |
| Ángulo nasolabial | < 90° o > 110° | Proyección nasal anormal |

## Disclaimer

⚠️ Este sistema tiene carácter orientativo y NO constituye un diagnóstico médico definitivo. Los resultados deben ser validados por un profesional médico calificado.
