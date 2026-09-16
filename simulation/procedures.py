"""
simulation/procedures.py — Procedimientos estéticos declarativos en mm / grados.

Cada procedimiento expone parámetros clínicos (ej. "Proyección de punta
+2 mm"). Cada parámetro se traduce en efectos sobre la malla MediaPipe:
un conjunto de vértices ancla que se desplazan en una dirección anatómica,
con caída gaussiana (sigma en mm) hacia los vértices vecinos y vértices
fijados que no se mueven (ej. el contorno del ojo al rellenar ojeras).

Marco facial (sin inclinación de cabeza): x → derecha de la imagen,
y → abajo, z → hacia atrás (MediaPipe). "forward" = hacia la cámara (-z).

Autor: FacialMetrics Pro
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulation.mesh import FACE_VERTEX_COUNT, FaceMesh3D


# ──────────────────────────────────────────────────────────────────────────────
# 1. Grupos de índices MediaPipe
# ──────────────────────────────────────────────────────────────────────────────

LIP_OUTER_UPPER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
LIP_OUTER_LOWER = (146, 91, 181, 84, 17, 314, 405, 321, 375)
LIP_INNER_UPPER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
LIP_INNER_LOWER = (95, 88, 178, 87, 14, 317, 402, 318, 324)
COMMISSURES = (61, 291)

LOWER_EYELIDS = (
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    263, 249, 390, 373, 374, 380, 381, 382, 362,
)
UPPER_EYELIDS = (
    246, 161, 160, 159, 158, 157, 173,
    466, 388, 387, 386, 385, 384, 398,
)
EYES = LOWER_EYELIDS + UPPER_EYELIDS

NOSE_TIP = (1, 4, 19, 45, 275)
NOSE_DORSUM = (197, 195, 5)
NOSE_ALAE = (129, 358, 49, 279, 64, 294, 98, 327, 102, 331, 48, 278)
SUBNASALE = 2
NOSE_BASE = (2, 94, 141, 370, 97, 326, 98, 327, 99, 328, 60, 290, 75, 305, 240, 460, 20, 250)
NASION = 168

CHIN = (152, 175, 199, 200, 377, 148, 176, 400)
GONIAL = (172, 136, 58, 215, 397, 365, 288, 435)
MALAR = (116, 117, 123, 50, 101, 345, 346, 352, 280, 330)
TEAR_TROUGH = (117, 118, 119, 120, 121, 346, 347, 348, 349, 350)
NASOLABIAL = (206, 216, 92, 203, 426, 436, 322, 423)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Modelo declarativo
# ──────────────────────────────────────────────────────────────────────────────

DIRECTIONS = {
    "forward": (0.0, 0.0, -1.0),
    "back": (0.0, 0.0, 1.0),
    "up": (0.0, -1.0, 0.0),
    "down": (0.0, 1.0, 0.0),
}


@dataclass(frozen=True)
class Effect:
    """Desplazamiento de un grupo de vértices ancla con caída gaussiana."""
    anchors: tuple[int, ...]
    direction: str                  # forward | back | up | down | lateral | tip_rotation
    sigma_mm: float = 5.0
    gain: float = 1.0
    pinned: tuple[int, ...] = ()
    side: str = ""                  # "" | "above_mouth" | "below_mouth"


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    unit: str                       # "mm" | "°"
    min_value: float
    max_value: float
    step: float
    effects: tuple[Effect, ...]
    help: str = ""
    ai_increase: str = ""           # Descripción para el prompt (valor > 0)
    ai_decrease: str = ""           # Descripción para el prompt (valor < 0)


@dataclass(frozen=True)
class Procedure:
    key: str
    name: str
    parameters: tuple[Parameter, ...]
    image: str = "rest"             # "rest" | "smile"
    note: str = ""


@dataclass(frozen=True)
class DentalOption:
    key: str
    label: str
    ai_prompt: str


@dataclass
class Preset:
    name: str
    description: str
    values: dict[str, float] = field(default_factory=dict)


def param_id(procedure: Procedure, parameter: Parameter) -> str:
    return f"{procedure.key}.{parameter.key}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Catálogo de procedimientos
# ──────────────────────────────────────────────────────────────────────────────

PROCEDURES: tuple[Procedure, ...] = (
    Procedure(
        key="nariz",
        name="Nariz (rinoplastia / rinomodelación)",
        parameters=(
            Parameter(
                "giba", "Reducción de giba dorsal", "mm", 0.0, 4.0, 0.5,
                (Effect(NOSE_DORSUM, "back", 4.0, pinned=(NASION, 1, 4)),),
                help="Rebaje del dorso nasal. Se aprecia en perfil y en el visor 3D.",
                ai_increase="reduce the nasal dorsal hump by about {v} mm, straighter nasal bridge",
            ),
            Parameter(
                "proyeccion", "Proyección de punta", "mm", -3.0, 3.0, 0.5,
                (Effect(NOSE_TIP, "forward", 3.5, pinned=(SUBNASALE, NASION, 6)),),
                ai_increase="increase nasal tip projection by about {v} mm",
                ai_decrease="reduce nasal tip projection by about {v} mm",
            ),
            Parameter(
                "rotacion", "Rotación de punta", "°", -10.0, 15.0, 1.0,
                (Effect(NOSE_TIP, "tip_rotation", 4.0, pinned=(SUBNASALE, NASION, 6)),),
                help="Positivo = rotación cefálica (punta hacia arriba).",
                ai_increase="rotate the nasal tip upwards by about {v} degrees",
                ai_decrease="rotate the nasal tip downwards by about {v} degrees",
            ),
            Parameter(
                "alar", "Ancho alar (total)", "mm", -4.0, 2.0, 0.5,
                (Effect(NOSE_ALAE, "lateral", 3.0, gain=0.5,
                        pinned=(1, 4, SUBNASALE, 195, 5)),),
                help="Negativo = alas más angostas.",
                ai_increase="widen the nasal alae by about {v} mm",
                ai_decrease="narrow the nasal alae by about {v} mm in total",
            ),
        ),
    ),
    Procedure(
        key="labios",
        name="Labios (volumen y contorno)",
        parameters=(
            Parameter(
                "superior", "Volumen labio superior", "mm", 0.0, 4.0, 0.5,
                (
                    Effect(LIP_OUTER_UPPER[2:-2], "up", 4.5,
                           pinned=LIP_INNER_UPPER + LIP_INNER_LOWER + COMMISSURES + NOSE_BASE,
                           side="above_mouth"),
                    Effect(LIP_OUTER_UPPER[2:-2], "forward", 4.5, gain=0.5,
                           pinned=COMMISSURES + NOSE_BASE, side="above_mouth"),
                ),
                help="Eversión del bermellón: la línea de cierre labial no se mueve.",
                ai_increase="fuller upper lip, vermilion about {v} mm taller, natural lip filler result",
            ),
            Parameter(
                "inferior", "Volumen labio inferior", "mm", 0.0, 4.0, 0.5,
                (
                    Effect(LIP_OUTER_LOWER[1:-1], "down", 4.5,
                           pinned=LIP_INNER_LOWER + LIP_INNER_UPPER + COMMISSURES,
                           side="below_mouth"),
                    Effect(LIP_OUTER_LOWER[1:-1], "forward", 4.5, gain=0.5,
                           pinned=COMMISSURES, side="below_mouth"),
                ),
                ai_increase="fuller lower lip, vermilion about {v} mm taller",
            ),
            Parameter(
                "comisuras", "Ancho de comisuras (total)", "mm", -3.0, 3.0, 0.5,
                (Effect(COMMISSURES, "lateral", 4.0, gain=0.5, pinned=(13, 14, 0, 17)),),
                ai_increase="slightly wider mouth, about {v} mm",
                ai_decrease="slightly narrower mouth, about {v} mm",
            ),
            Parameter(
                "elevacion", "Elevación de comisuras", "mm", 0.0, 2.0, 0.25,
                (Effect(COMMISSURES, "up", 4.0, pinned=(13, 14, 0, 17)),),
                ai_increase="lift the mouth corners by about {v} mm",
            ),
        ),
    ),
    Procedure(
        key="menton",
        name="Mentón (mentoplastia / relleno)",
        parameters=(
            Parameter(
                "avance", "Avance de mentón", "mm", -3.0, 6.0, 0.5,
                (Effect(CHIN, "forward", 7.0, pinned=LIP_INNER_LOWER + LIP_OUTER_LOWER,
                        side="below_mouth"),),
                help="Se aprecia en perfil y en el visor 3D.",
                ai_increase="more projected chin, advanced about {v} mm",
                ai_decrease="less projected chin, about {v} mm",
            ),
            Parameter(
                "vertical", "Alargamiento vertical", "mm", -4.0, 6.0, 0.5,
                (Effect(CHIN + (149, 378), "down", 6.0,
                        pinned=LIP_INNER_LOWER + LIP_OUTER_LOWER, side="below_mouth"),),
                ai_increase="chin about {v} mm longer vertically",
                ai_decrease="chin about {v} mm shorter vertically",
            ),
        ),
    ),
    Procedure(
        key="mandibula",
        name="Mandíbula / perfilado",
        parameters=(
            Parameter(
                "gonial", "Ancho del ángulo mandibular (por lado)", "mm", -5.0, 5.0, 0.5,
                (Effect(GONIAL, "lateral", 8.0,
                        pinned=(152, 175) + COMMISSURES + (234, 454)),),
                help="Positivo = relleno de ángulo; negativo = reducción de masetero.",
                ai_increase="more defined, wider jaw angles (about {v} mm per side), sharp jawline",
                ai_decrease="slimmer lower face, masseter reduction about {v} mm per side",
            ),
            Parameter(
                "definicion", "Definición del borde mandibular", "mm", 0.0, 3.0, 0.5,
                (Effect(GONIAL + (149, 150, 176, 378, 379, 400), "forward", 6.0,
                        pinned=(152,) + COMMISSURES),),
                ai_increase="well defined mandibular border and jawline",
            ),
        ),
    ),
    Procedure(
        key="rellenos",
        name="Rellenos (ácido hialurónico)",
        parameters=(
            Parameter(
                "pomulo", "Proyección malar", "mm", 0.0, 4.0, 0.5,
                (
                    Effect(MALAR, "forward", 7.0, pinned=EYES + (129, 358)),
                    Effect(MALAR, "lateral", 7.0, gain=0.3, pinned=EYES + (129, 358)),
                    Effect(MALAR, "up", 7.0, gain=0.2, pinned=EYES + (129, 358)),
                ),
                ai_increase="enhanced cheekbone projection with dermal filler, about {v} mm",
            ),
            Parameter(
                "ojeras", "Relleno de ojeras", "mm", 0.0, 3.0, 0.5,
                (Effect(TEAR_TROUGH, "forward", 4.0, pinned=EYES),),
                ai_increase="filled tear troughs, softer under-eye hollows, less dark circles",
            ),
            Parameter(
                "surco", "Surco nasogeniano", "mm", 0.0, 3.0, 0.5,
                (Effect(NASOLABIAL, "forward", 4.0, pinned=COMMISSURES + (129, 358)),),
                ai_increase="softened nasolabial folds",
            ),
        ),
    ),
    Procedure(
        key="sonrisa",
        name="Sonrisa (exposición gingival)",
        image="smile",
        note="Usa la foto frontal en sonrisa.",
        parameters=(
            Parameter(
                "gingival", "Reducción de exposición gingival", "mm", 0.0, 4.0, 0.5,
                (Effect(LIP_OUTER_UPPER[1:-1] + LIP_INNER_UPPER[1:-1], "down", 6.0,
                        pinned=NOSE_BASE, side="above_mouth"),),
                help="Descenso del labio superior (toxina / reposición labial).",
                ai_increase="upper lip lowered about {v} mm when smiling, less gum showing",
            ),
        ),
    ),
)

DENTAL_OPTIONS: tuple[DentalOption, ...] = (
    DentalOption("blanqueamiento", "Blanqueamiento dental",
                 "natural whiter teeth (professional whitening, not artificial)"),
    DentalOption("alineacion", "Alineación dental",
                 "straight, well aligned teeth with a harmonious smile line"),
    DentalOption("carillas", "Carillas / diseño de sonrisa",
                 "natural looking porcelain veneers with ideal tooth proportions"),
)

PRESETS: tuple[Preset, ...] = (
    Preset("Rinoplastia sutil", "Rebaje de giba, leve rotación y alas más angostas",
           {"nariz.giba": 2.0, "nariz.rotacion": 5.0, "nariz.alar": -1.0}),
    Preset("Perfilado de labios", "Volumen natural y comisuras elevadas",
           {"labios.superior": 1.5, "labios.inferior": 1.0, "labios.elevacion": 0.5}),
    Preset("Mentoplastia de avance", "Mentón más proyectado y levemente más largo",
           {"menton.avance": 4.0, "menton.vertical": 1.0}),
    Preset("Perfilado mandibular", "Ángulos y borde mandibular definidos",
           {"mandibula.gonial": 2.0, "mandibula.definicion": 1.5, "menton.avance": 2.0}),
    Preset("Rejuvenecimiento tercio medio", "Pómulos, ojeras y surcos",
           {"rellenos.pomulo": 2.0, "rellenos.ojeras": 1.5, "rellenos.surco": 1.5}),
    Preset("Corrección de sonrisa gingival", "Descenso del labio superior en sonrisa",
           {"sonrisa.gingival": 2.5}),
)


def all_parameters() -> list[tuple[Procedure, Parameter]]:
    return [(proc, par) for proc in PROCEDURES for par in proc.parameters]


def default_values() -> dict[str, float]:
    return {param_id(proc, par): 0.0 for proc, par in all_parameters()}


# ──────────────────────────────────────────────────────────────────────────────
# 4. Cálculo de desplazamientos
# ──────────────────────────────────────────────────────────────────────────────

def compute_displacement(
    mesh: FaceMesh3D,
    values: dict[str, float],
    pixel_per_mm: float,
    image: str = "rest",
) -> np.ndarray:
    """
    Desplazamiento (N, 3) en px de todos los vértices para los valores dados.

    Args:
        mesh: Malla en reposo.
        values: {"procedimiento.parametro": valor en mm o grados}.
        pixel_per_mm: Escala de calibración.
        image: Foto sobre la que se simula ("rest" | "smile"). Los
            procedimientos exclusivos de sonrisa solo aplican con "smile".
    """
    if pixel_per_mm <= 0:
        raise ValueError("Se requiere calibración (px/mm > 0) para simular.")

    frame = mesh.to_face_frame(mesh.vertices)
    midline = mesh.midline_x()
    disp = np.zeros_like(mesh.vertices)

    for proc, par in all_parameters():
        if proc.image == "smile" and image != "smile":
            continue
        value = float(values.get(param_id(proc, par), 0.0))
        if abs(value) < 1e-9:
            continue
        for effect in par.effects:
            weights = _effect_weights(frame, effect, pixel_per_mm)
            disp += _effect_vectors(frame, effect, value, pixel_per_mm, midline) * weights[:, None]

    return mesh.from_face_frame_vectors(disp)


def _effect_weights(frame: np.ndarray, effect: Effect, pixel_per_mm: float) -> np.ndarray:
    xy = frame[:, :2]
    anchors = xy[list(effect.anchors)]
    d2 = ((xy[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2) / pixel_per_mm ** 2
    weights = np.exp(-d2 / (2.0 * effect.sigma_mm ** 2)).max(axis=1)
    weights[weights < 0.02] = 0.0
    weights[FACE_VERTEX_COUNT:] = 0.0       # iris: sigue al párpado vía warp
    weights[list(effect.pinned)] = 0.0
    if effect.side:
        mouth_line = np.interp(frame[:, 0], *_mouth_line(frame))
        tolerance = 0.5     # px: incluye los puntos sobre la línea de cierre
        if effect.side == "above_mouth":
            weights[frame[:, 1] > mouth_line + tolerance] = 0.0
        else:
            weights[frame[:, 1] < mouth_line - tolerance] = 0.0
    return weights


def _mouth_line(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Línea de cierre labial (borde interno del labio superior), ordenada en x."""
    pts = frame[list(LIP_INNER_UPPER)]
    order = np.argsort(pts[:, 0])
    return pts[order, 0], pts[order, 1]


def _effect_vectors(
    frame: np.ndarray,
    effect: Effect,
    value: float,
    pixel_per_mm: float,
    midline: float,
) -> np.ndarray:
    n = len(frame)

    if effect.direction == "tip_rotation":
        theta = np.radians(value) * effect.gain
        pivot = frame[SUBNASALE]
        y = frame[:, 1] - pivot[1]
        z = frame[:, 2] - pivot[2]
        vectors = np.zeros((n, 3))
        vectors[:, 1] = (y * np.cos(theta) + z * np.sin(theta)) - y
        vectors[:, 2] = (-y * np.sin(theta) + z * np.cos(theta)) - z
        return vectors

    magnitude = value * effect.gain * pixel_per_mm

    if effect.direction == "lateral":
        offset = frame[:, 0] - midline
        # Atenuar cerca de la línea media para que no haya salto de signo
        side = np.clip(offset / (3.0 * pixel_per_mm), -1.0, 1.0)
        vectors = np.zeros((n, 3))
        vectors[:, 0] = side * magnitude
        return vectors

    return np.tile(np.array(DIRECTIONS[effect.direction]) * magnitude, (n, 1))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Descripciones
# ──────────────────────────────────────────────────────────────────────────────

def active_changes(values: dict[str, float]) -> list[tuple[Procedure, Parameter, float]]:
    return [
        (proc, par, values[param_id(proc, par)])
        for proc, par in all_parameters()
        if abs(values.get(param_id(proc, par), 0.0)) > 1e-9
    ]


def describe_for_ai(values: dict[str, float], dental: list[str] | None = None) -> list[str]:
    """Descripciones en inglés de los cambios activos, para el prompt de IA."""
    lines = []
    for _, par, value in active_changes(values):
        template = par.ai_increase if value > 0 else (par.ai_decrease or par.ai_increase)
        if template:
            lines.append(template.format(v=f"{abs(value):g}"))
    dental_by_key = {opt.key: opt for opt in DENTAL_OPTIONS}
    for key in dental or []:
        lines.append(dental_by_key[key].ai_prompt)
    return lines
