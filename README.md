# FacialMetrics Pro 🔬

Sistema de análisis facial cefalométrico/antropométrico automatizado para clínicas de estética facial y odontológica.

## Descripción

El sistema procesa fotos médicas estandarizadas (frontal en reposo, frontal en sonrisa y perfil estricto) para:

- **Detectar puntos anatómicos** (landmarks) con MediaPipe Face Mesh (478 puntos)
- **Calibrar escala** píxeles → milímetros (manual con regla, o automático por distancia intercantal)
- **Calcular proporciones faciales**: Ley de Tercios, Ley de Quintos, Línea Media, Análisis de Perfil
- **Emitir diagnósticos preliminares** con motor de reglas clínicas (alertas de asimetría, sonrisa gingival, etc.)
- **Simular resultados estéticos** en mm sobre la malla 3D de MediaPipe (nariz, labios, mentón, mandíbula, rellenos, sonrisa gingival), con re-medición, visor 3D y refinado fotorrealista opcional con IA vía OpenRouter
- **Generar reportes PDF** profesionales con imágenes anotadas, tablas de mediciones y alertas clínicas

## Stack Tecnológico

| Componente | Tecnología |
|:---|:---|
| Procesamiento de imagen | OpenCV, MediaPipe, NumPy, Pillow |
| Cálculos científicos | SciPy (Delaunay) |
| Visor 3D | three.js (CDN) |
| IA generativa (opcional) | OpenRouter (modelos de imagen), fal.ai: Rodin V2.5 / Hunyuan3D Pro / TRELLIS (modelos 3D) |
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
├── simulation/            # Simulación antes/después
│   ├── mesh.py            #   Malla 3D (478 landmarks + teselación)
│   ├── procedures.py      #   Procedimientos y presets en mm / grados
│   ├── warp.py            #   Warp piecewise-affine de la foto
│   ├── pipeline.py        #   Simulación + re-medición
│   ├── viewer3d.py        #   Visor 3D antes/después (three.js)
│   ├── ai_refine.py       #   Refinado con IA (OpenRouter) + verificación
│   ├── model3d.py         #   Cabeza 3D completa con fal.ai (Rodin / Hunyuan3D / TRELLIS)
│   └── viewer_glb.py      #   Visor 3D de modelos GLB antes/después
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

### 4. (Opcional) Configurar OpenRouter

Para el refinado fotorrealista, definí la API key de una de estas formas
(o pegala en la barra lateral de la app, solo para la sesión):

```bash
export OPENROUTER_API_KEY=sk-or-...
# o en .streamlit/secrets.toml:  OPENROUTER_API_KEY = "sk-or-..."
```

Para la cabeza 3D completa, lo mismo con `FAL_KEY` (generada en fal.ai, pago
por uso). Los modelos generados se guardan en `model3d_cache/` (fuera de git)
y no se vuelven a cobrar si las fotos no cambian.

### 5. Ejecutar tests

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
5. **🔄 Simulación**: Elegir un preset o ajustar parámetros en mm/grados; ver antes/después, la re-medición y el visor 3D (frente, 3/4, perfil). Opcionalmente refinar con IA (requiere consentimiento del paciente y API key de OpenRouter).
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
