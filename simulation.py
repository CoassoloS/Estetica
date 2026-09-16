"""
simulation.py — Módulo de Simulación Visual (Antes / Después).

Implementa:
    1. Deformación elástica localizada con Thin Plate Splines (TPS)
    2. Stub de API para Inpainting (FLUX Fill / SDXL)

Permite simular cambios estéticos en nariz, mentón y labios sobre la
imagen del paciente de forma interactiva.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from landmarks import AnthropometricPoint, LandmarkResult


# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuración de Simulación
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WarpPoint:
    """Punto de control para la deformación."""
    original: tuple[int, int]       # Posición original (x, y)
    displaced: tuple[int, int]      # Posición objetivo (x, y)
    region: str = "nose"            # "nose" | "chin" | "lips"


@dataclass
class SimulationConfig:
    """Configuración de la simulación visual."""
    warp_radius: int = 80           # Radio de influencia del warp en px
    smoothness: float = 50.0        # Suavidad de la interpolación TPS
    blend_border: int = 20          # Píxeles de borde para blending suave


# ──────────────────────────────────────────────────────────────────────────────
# 2. Deformación con Thin Plate Splines (TPS) via RBF
# ──────────────────────────────────────────────────────────────────────────────

class MeshWarper:
    """
    Deformación elástica localizada usando Radial Basis Functions (RBF).

    Implementa un warp suave tipo TPS que desplaza puntos de control
    y deforma la imagen de forma continua y diferenciable alrededor
    de ellos, manteniendo el resto de la imagen intacto.
    """

    def __init__(self, config: SimulationConfig = SimulationConfig()) -> None:
        self.config = config

    def warp(
        self,
        image: np.ndarray,
        warp_points: list[WarpPoint],
        mask_region: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Aplicar deformación elástica a la imagen.

        Args:
            image: Imagen BGR de entrada.
            warp_points: Lista de puntos de control con desplazamiento.
            mask_region: Máscara opcional que limita la zona de deformación.

        Returns:
            Imagen deformada.
        """
        if not warp_points:
            return image.copy()

        h, w = image.shape[:2]

        # Preparar puntos de control
        src_pts = np.array(
            [wp.original for wp in warp_points], dtype=np.float64
        )
        dst_pts = np.array(
            [wp.displaced for wp in warp_points], dtype=np.float64
        )

        # Calcular desplazamientos
        displacements = dst_pts - src_pts  # (N, 2)

        # Si no hay desplazamiento significativo, retornar copia
        if np.max(np.abs(displacements)) < 0.5:
            return image.copy()

        # Añadir puntos de anclaje en los bordes para estabilizar
        border_anchors = self._generate_border_anchors(w, h, src_pts)
        all_src = np.vstack([src_pts, border_anchors])
        all_disp = np.vstack([
            displacements,
            np.zeros((len(border_anchors), 2)),
        ])

        # Interpolar con RBF (equivalente a TPS)
        rbf_x = RBFInterpolator(
            all_src,
            all_disp[:, 0],
            kernel="thin_plate_spline",
            smoothing=self.config.smoothness,
        )
        rbf_y = RBFInterpolator(
            all_src,
            all_disp[:, 1],
            kernel="thin_plate_spline",
            smoothing=self.config.smoothness,
        )

        # Generar mapa de remap
        # Para eficiencia, solo calcular en la zona de interés
        bbox = self._compute_bounding_box(
            src_pts, dst_pts, w, h, self.config.warp_radius
        )
        x_min, y_min, x_max, y_max = bbox

        # Crear grilla de puntos en la zona de interés
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Calcular desplazamientos en la grilla
        dx = rbf_x(grid_points).reshape(yy.shape)
        dy = rbf_y(grid_points).reshape(yy.shape)

        # Mapa de remap completo
        map_x = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)
        map_y = np.arange(h, dtype=np.float32)[:, np.newaxis].repeat(w, axis=1)

        # Aplicar desplazamiento inverso (para remap)
        map_x[y_min:y_max, x_min:x_max] -= dx.astype(np.float32)
        map_y[y_min:y_max, x_min:x_max] -= dy.astype(np.float32)

        # Aplicar remap
        result = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Blending suave en los bordes de la zona
        if mask_region is not None:
            mask_float = mask_region.astype(np.float32) / 255.0
            mask_float = cv2.GaussianBlur(
                mask_float,
                (self.config.blend_border * 2 + 1,) * 2,
                0,
            )
            if len(mask_float.shape) == 2:
                mask_float = mask_float[:, :, np.newaxis]

            result = (
                result.astype(np.float32) * mask_float
                + image.astype(np.float32) * (1 - mask_float)
            ).astype(np.uint8)

        return result

    def _generate_border_anchors(
        self,
        w: int,
        h: int,
        src_pts: np.ndarray,
        n_per_edge: int = 8,
    ) -> np.ndarray:
        """Generar puntos de anclaje en los bordes de la imagen."""
        anchors = []

        # Bordes
        for i in range(n_per_edge):
            t = i / (n_per_edge - 1)
            anchors.extend([
                [t * w, 0],            # Top
                [t * w, h - 1],        # Bottom
                [0, t * h],            # Left
                [w - 1, t * h],        # Right
            ])

        return np.array(anchors, dtype=np.float64)

    def _compute_bounding_box(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        w: int,
        h: int,
        padding: int,
    ) -> tuple[int, int, int, int]:
        """Calcular bounding box de la zona de deformación."""
        all_pts = np.vstack([src_pts, dst_pts])

        x_min = max(0, int(np.min(all_pts[:, 0]) - padding))
        y_min = max(0, int(np.min(all_pts[:, 1]) - padding))
        x_max = min(w, int(np.max(all_pts[:, 0]) + padding))
        y_max = min(h, int(np.max(all_pts[:, 1]) + padding))

        return (x_min, y_min, x_max, y_max)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Generador de Puntos de Warp desde Landmarks
# ──────────────────────────────────────────────────────────────────────────────

def generate_nose_warp_points(
    landmarks: LandmarkResult,
    dx: int = 0,
    dy: int = 0,
    scale: float = 1.0,
) -> list[WarpPoint]:
    """
    Generar puntos de warp para simulación de rinoplastia.

    Args:
        landmarks: Landmarks faciales detectados.
        dx: Desplazamiento horizontal de la punta nasal (px).
        dy: Desplazamiento vertical de la punta nasal (px).
        scale: Factor de escala nasal (1.0 = sin cambio, < 1.0 = reducción).

    Returns:
        Lista de WarpPoints para la deformación nasal.
    """
    AP = AnthropometricPoint
    points = []

    pn = landmarks.get(AP.PRONASALE)
    sn = landmarks.get(AP.SUBNASALE)
    al_l = landmarks.get(AP.ALAR_L)
    al_r = landmarks.get(AP.ALAR_R)
    nb = landmarks.get(AP.NASAL_BRIDGE)

    # Centro nasal para escala
    center_x = (al_l[0] + al_r[0]) // 2
    center_y = (nb[1] + sn[1]) // 2

    def scale_point(
        pt: tuple[int, int], cx: int, cy: int, s: float
    ) -> tuple[int, int]:
        return (
            int(cx + (pt[0] - cx) * s),
            int(cy + (pt[1] - cy) * s),
        )

    # Punta nasal: desplazamiento + escala
    pn_new = scale_point(pn, center_x, center_y, scale)
    pn_new = (pn_new[0] + dx, pn_new[1] + dy)
    points.append(WarpPoint(original=pn, displaced=pn_new, region="nose"))

    # Alas nasales: solo escala
    al_l_new = scale_point(al_l, center_x, center_y, scale)
    al_r_new = scale_point(al_r, center_x, center_y, scale)
    points.append(WarpPoint(original=al_l, displaced=al_l_new, region="nose"))
    points.append(WarpPoint(original=al_r, displaced=al_r_new, region="nose"))

    # Subnasale: escala vertical
    sn_new = scale_point(sn, center_x, center_y, scale)
    sn_new = (sn_new[0], sn_new[1] + dy // 2)
    points.append(WarpPoint(original=sn, displaced=sn_new, region="nose"))

    return points


def generate_chin_warp_points(
    landmarks: LandmarkResult,
    dx: int = 0,
    dy: int = 0,
) -> list[WarpPoint]:
    """
    Generar puntos de warp para simulación de mentoplastia.

    Args:
        landmarks: Landmarks faciales detectados.
        dx: Desplazamiento horizontal del mentón (px). Positivo = avance.
        dy: Desplazamiento vertical del mentón (px). Positivo = alargamiento.

    Returns:
        Lista de WarpPoints para la deformación del mentón.
    """
    AP = AnthropometricPoint
    points = []

    pog = landmarks.get(AP.POGONION)
    me = landmarks.get(AP.MENTON)
    b = landmarks.get(AP.SOFT_TISSUE_B)

    # Pogonion: desplazamiento completo
    points.append(WarpPoint(
        original=pog,
        displaced=(pog[0] + dx, pog[1] + dy),
        region="chin",
    ))

    # Menton: desplazamiento completo
    points.append(WarpPoint(
        original=me,
        displaced=(me[0] + dx, me[1] + dy),
        region="chin",
    ))

    # Punto B: desplazamiento parcial (gradiente suave)
    points.append(WarpPoint(
        original=b,
        displaced=(b[0] + dx // 2, b[1] + dy // 3),
        region="chin",
    ))

    return points


def generate_lips_warp_points(
    landmarks: LandmarkResult,
    upper_dy: int = 0,
    lower_dy: int = 0,
    width_scale: float = 1.0,
) -> list[WarpPoint]:
    """
    Generar puntos de warp para simulación labial.

    Args:
        landmarks: Landmarks faciales detectados.
        upper_dy: Desplazamiento vertical labio superior (negativo = volumen).
        lower_dy: Desplazamiento vertical labio inferior (positivo = volumen).
        width_scale: Escala horizontal de comisuras (1.0 = sin cambio).

    Returns:
        Lista de WarpPoints para la deformación labial.
    """
    AP = AnthropometricPoint
    points = []

    sts = landmarks.get(AP.STOMION_SUPERIOR)
    sti = landmarks.get(AP.STOMION_INFERIOR)
    ch_l = landmarks.get(AP.CHEILION_L)
    ch_r = landmarks.get(AP.CHEILION_R)

    center_x = (ch_l[0] + ch_r[0]) // 2

    # Labio superior
    points.append(WarpPoint(
        original=sts,
        displaced=(sts[0], sts[1] + upper_dy),
        region="lips",
    ))

    # Labio inferior
    points.append(WarpPoint(
        original=sti,
        displaced=(sti[0], sti[1] + lower_dy),
        region="lips",
    ))

    # Comisuras: escala horizontal
    ch_l_new = (int(center_x + (ch_l[0] - center_x) * width_scale), ch_l[1])
    ch_r_new = (int(center_x + (ch_r[0] - center_x) * width_scale), ch_r[1])
    points.append(WarpPoint(original=ch_l, displaced=ch_l_new, region="lips"))
    points.append(WarpPoint(original=ch_r, displaced=ch_r_new, region="lips"))

    return points


# ──────────────────────────────────────────────────────────────────────────────
# 4. Máscara de Región
# ──────────────────────────────────────────────────────────────────────────────

def create_region_mask(
    image_shape: tuple[int, int],
    landmarks: LandmarkResult,
    region: str,
    padding: int = 30,
) -> np.ndarray:
    """
    Crear máscara binaria para una región facial.

    Args:
        image_shape: (height, width) de la imagen.
        landmarks: Landmarks detectados.
        region: "nose" | "chin" | "lips".
        padding: Píxeles de padding alrededor de la región.

    Returns:
        Máscara binaria (0-255) del tamaño de la imagen.
    """
    AP = AnthropometricPoint
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if region == "nose":
        pts = [
            landmarks.get(AP.NASAL_BRIDGE),
            landmarks.get(AP.ALAR_L),
            landmarks.get(AP.PRONASALE),
            landmarks.get(AP.SUBNASALE),
            landmarks.get(AP.ALAR_R),
        ]
    elif region == "chin":
        pts = [
            landmarks.get(AP.STOMION_INFERIOR),
            landmarks.get(AP.POGONION),
            landmarks.get(AP.MENTON),
        ]
        # Expandir lateralmente
        ch_l = landmarks.get(AP.CHEILION_L)
        ch_r = landmarks.get(AP.CHEILION_R)
        me = landmarks.get(AP.MENTON)
        pts = [
            (ch_l[0], landmarks.get(AP.STOMION_INFERIOR)[1]),
            (ch_r[0], landmarks.get(AP.STOMION_INFERIOR)[1]),
            (ch_r[0], me[1] + padding),
            (ch_l[0], me[1] + padding),
        ]
    elif region == "lips":
        ch_l = landmarks.get(AP.CHEILION_L)
        ch_r = landmarks.get(AP.CHEILION_R)
        sts = landmarks.get(AP.STOMION_SUPERIOR)
        sti = landmarks.get(AP.STOMION_INFERIOR)
        pts = [
            (ch_l[0] - padding, sts[1] - padding),
            (ch_r[0] + padding, sts[1] - padding),
            (ch_r[0] + padding, sti[1] + padding),
            (ch_l[0] - padding, sti[1] + padding),
        ]
    else:
        return mask

    # Dibujar polígono relleno
    pts_array = np.array(pts, dtype=np.int32)
    cv2.fillConvexPoly(mask, pts_array, 255)

    # Suavizar bordes
    mask = cv2.GaussianBlur(mask, (padding * 2 + 1, padding * 2 + 1), 0)

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# 5. Stub de API para Inpainting
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InpaintingRequest:
    """Payload para un endpoint de Inpainting (FLUX Fill / SDXL)."""
    image_base64: str           # Imagen original en base64
    mask_base64: str            # Máscara de la zona a modificar en base64
    prompt: str                 # Prompt de modificación
    negative_prompt: str = ""
    strength: float = 0.75      # Fuerza de la transformación (0-1)
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    seed: int = -1              # -1 = aleatorio
    preserve_identity: bool = True  # Flag para preservar identidad del paciente


def prepare_inpainting_request(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    strength: float = 0.75,
) -> InpaintingRequest:
    """
    Preparar un request de Inpainting para enviar a un endpoint externo.

    Args:
        image: Imagen BGR original.
        mask: Máscara binaria de la zona a modificar.
        prompt: Descripción del cambio deseado.
        strength: Fuerza de la transformación.

    Returns:
        InpaintingRequest con los datos codificados en base64.
    """
    import base64

    # Codificar imagen
    _, img_buffer = cv2.imencode(".png", image)
    img_b64 = base64.b64encode(img_buffer).decode("utf-8")

    # Codificar máscara
    _, mask_buffer = cv2.imencode(".png", mask)
    mask_b64 = base64.b64encode(mask_buffer).decode("utf-8")

    return InpaintingRequest(
        image_base64=img_b64,
        mask_base64=mask_b64,
        prompt=prompt,
        strength=strength,
    )


def send_inpainting_request(
    request: InpaintingRequest,
    endpoint_url: str = "http://localhost:8080/inpaint",
) -> Optional[np.ndarray]:
    """
    Enviar request de Inpainting a un endpoint externo.

    NOTA: Este es un stub. En producción, reemplazar con la llamada
    real al endpoint de FLUX Fill, SDXL Inpainting, o similar.

    Args:
        request: Payload del request.
        endpoint_url: URL del endpoint de Inpainting.

    Returns:
        Imagen resultante o None si el endpoint no está disponible.
    """
    try:
        import requests
        import base64

        payload = {
            "image": request.image_base64,
            "mask": request.mask_base64,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "strength": request.strength,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "seed": request.seed,
            "preserve_identity": request.preserve_identity,
        }

        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=120,
        )

        if response.status_code == 200:
            result_b64 = response.json().get("image")
            if result_b64:
                img_bytes = base64.b64decode(result_b64)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return None

    except Exception:
        # Endpoint no disponible — retornar None silenciosamente
        return None
