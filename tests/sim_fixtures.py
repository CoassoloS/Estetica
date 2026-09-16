"""
sim_fixtures.py — Malla facial sintética para los tests de simulación.

Los 478 vértices se ubican al azar (semilla fija) dentro de un óvalo facial
y los índices relevantes (puntos antropométricos, labios, nariz) se fijan
en posiciones controladas, con proporciones de la cara "ideal" de
test_measurements.py.
"""

import numpy as np

from landmarks import AnthropometricPoint, LandmarkResult, MEDIAPIPE_INDEX_MAP

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 500

IDEAL_POINTS = {
    AnthropometricPoint.GLABELLA: (200, 150),
    AnthropometricPoint.NASION: (200, 160),
    AnthropometricPoint.NASAL_BRIDGE: (200, 180),
    AnthropometricPoint.PRONASALE: (200, 220),
    AnthropometricPoint.SUBNASALE: (200, 250),
    AnthropometricPoint.STOMION_SUPERIOR: (200, 280),
    AnthropometricPoint.STOMION_INFERIOR: (200, 290),
    AnthropometricPoint.POGONION: (200, 340),
    AnthropometricPoint.MENTON: (200, 350),
    AnthropometricPoint.ENDOCANTHION_L: (170, 160),
    AnthropometricPoint.ENDOCANTHION_R: (230, 160),
    AnthropometricPoint.EXOCANTHION_L: (110, 160),
    AnthropometricPoint.EXOCANTHION_R: (290, 160),
    AnthropometricPoint.TRAGION_L: (50, 160),
    AnthropometricPoint.TRAGION_R: (350, 160),
    AnthropometricPoint.CHEILION_L: (170, 285),
    AnthropometricPoint.CHEILION_R: (230, 285),
    AnthropometricPoint.ALAR_L: (185, 230),
    AnthropometricPoint.ALAR_R: (215, 230),
    AnthropometricPoint.SOFT_TISSUE_A: (200, 255),
    AnthropometricPoint.SOFT_TISSUE_B: (200, 310),
}

LIP_INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIP_INNER_LOWER = [95, 88, 178, 87, 14, 317, 402, 318, 324]
LIP_OUTER_UPPER = [185, 40, 39, 37, 0, 267, 269, 270, 409]
LIP_OUTER_LOWER = [146, 91, 181, 84, 17, 314, 405, 321, 375]


def make_synthetic_vertices(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, 478)
    radius = np.sqrt(rng.uniform(0, 1, 478))
    x = 200 + 140 * radius * np.cos(angle)
    y = 250 + 190 * radius * np.sin(angle)
    z = -20.0 * (1 - radius ** 2)
    vertices = np.column_stack([x, y, z])

    for point, (px, py) in IDEAL_POINTS.items():
        vertices[MEDIAPIPE_INDEX_MAP[point]] = (px, py, -20.0)

    vertices[4, 2] = -40.0                     # Punta nasal proyectada
    vertices[2, 2] = -25.0                     # Subnasale
    vertices[1] = (200, 225, -38.0)
    vertices[19] = (200, 238, -32.0)
    vertices[45] = (194, 226, -34.0)
    vertices[275] = (206, 226, -34.0)

    for i, idx in enumerate(LIP_INNER_UPPER):
        vertices[idx] = (172 + i * 5.6, 283, -22.0)
    vertices[13] = (200, 280, -22.0)
    for i, idx in enumerate(LIP_INNER_LOWER):
        vertices[idx] = (178 + i * 5.5, 287, -22.0)
    vertices[14] = (200, 290, -22.0)
    for i, idx in enumerate(LIP_OUTER_UPPER):
        vertices[idx] = (176 + i * 6.0, 272, -23.0)
    for i, idx in enumerate(LIP_OUTER_LOWER):
        vertices[idx] = (176 + i * 6.0, 300, -23.0)

    return vertices


def make_synthetic_landmarks(seed: int = 7) -> LandmarkResult:
    vertices = make_synthetic_vertices(seed)
    points = {
        point: (int(round(vertices[idx, 0])), int(round(vertices[idx, 1])))
        for point, idx in MEDIAPIPE_INDEX_MAP.items()
        if idx is not None
    }
    points[AnthropometricPoint.TRICHION] = (200, 50)
    return LandmarkResult(
        points=points,
        raw_landmarks=[tuple(v) for v in vertices],
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        detection_confidence=0.5,
        estimated_points=[AnthropometricPoint.TRICHION],
    )


def make_texture_image(seed: int = 3) -> np.ndarray:
    """Imagen suave con textura (para poder rastrear el movimiento de píxeles)."""
    import cv2

    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.5)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return (smooth * 255).astype(np.uint8)
