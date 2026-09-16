"""
simulation — Simulación estética antes/después.

Pipeline híbrido:
    1. Malla 3D de MediaPipe (`mesh`)
    2. Procedimientos en mm / grados sobre la malla (`procedures`)
    3. Warp piecewise-affine de la foto (`warp`) + re-medición (`pipeline`)
    4. Visor 3D antes/después (`viewer3d`)
    5. Refinado fotorrealista opcional vía OpenRouter (`ai_refine`)
    6. Cabeza 3D completa generativa con fal.ai: Rodin / Hunyuan3D / TRELLIS (`model3d`, `viewer_glb`)

Autor: FacialMetrics Pro
"""

from simulation.ai_refine import (
    DEFAULT_MODEL,
    OpenRouterBackend,
    RefineError,
    VerificationResult,
    build_prompt,
    list_image_models,
    verify_geometry,
)
from simulation.mesh import FaceMesh3D
from simulation.pipeline import MeasurementDelta, SimulationOutcome, run_simulation
from simulation.procedures import (
    DENTAL_OPTIONS,
    PRESETS,
    PROCEDURES,
    active_changes,
    default_values,
    describe_for_ai,
    param_id,
)
from simulation.model3d import DEFAULT_MODEL_KEY, FAL_MODELS, FalClient, Model3DError
from simulation.viewer3d import build_viewer_html
from simulation.viewer_glb import build_glb_viewer_html
from simulation.warp import warp_image

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_KEY",
    "DENTAL_OPTIONS",
    "FaceMesh3D",
    "FalClient",
    "MeasurementDelta",
    "Model3DError",
    "OpenRouterBackend",
    "PRESETS",
    "PROCEDURES",
    "FAL_MODELS",
    "RefineError",
    "SimulationOutcome",
    "VerificationResult",
    "active_changes",
    "build_glb_viewer_html",
    "build_prompt",
    "build_viewer_html",
    "default_values",
    "describe_for_ai",
    "list_image_models",
    "param_id",
    "run_simulation",
    "verify_geometry",
    "warp_image",
]
