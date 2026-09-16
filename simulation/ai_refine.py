"""
simulation/ai_refine.py — Refinado fotorrealista con IA generativa (OpenRouter).

La geometría la define el warp en mm; la IA solo vuelve realista el
resultado (sombras, volumen, piel, dientes). Luego se verifica que la
salida respete la geometría simulada re-detectando los landmarks.

OpenRouter expone dos formas de generar imágenes; se usa la API de
imágenes (`/api/v1/images` con `input_references`) y, si el modelo no la
soporta, chat completions con `modalities: ["image", "text"]`.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np
import requests

from simulation.mesh import FACE_VERTEX_COUNT, FaceMesh3D


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash-image"
REQUEST_TIMEOUT_S = 180
VERIFICATION_THRESHOLD_MM = 1.5


class RefineError(RuntimeError):
    """Error del proveedor de IA, con mensaje apto para mostrar al usuario."""


class RefineBackend(Protocol):
    def refine(self, original: np.ndarray, guide: np.ndarray, prompt: str) -> np.ndarray: ...


# ──────────────────────────────────────────────────────────────────────────────
# 1. Prompt
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(changes: list[str]) -> str:
    bullet_list = "\n".join(f"- {c}" for c in changes) or "- no changes"
    return (
        "You are retouching a clinical portrait for an aesthetic medicine consultation.\n"
        "Image 1 is the original patient photo. Image 2 is a geometric simulation of the "
        "planned treatment (it may look slightly stretched).\n"
        "Produce a single photorealistic photograph of the same person showing the planned "
        "result, matching the facial geometry of image 2.\n"
        f"Planned changes:\n{bullet_list}\n"
        "Strict rules: keep the exact same identity, pose, framing, image size and aspect "
        "ratio, facial expression, eyes, skin tone and texture, hair, lighting, background "
        "and clothing. Do not beautify, smooth skin, add makeup or change anything that is "
        "not listed. The result must look like an untouched real photograph."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cliente OpenRouter
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenRouterBackend:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = OPENROUTER_BASE_URL

    def refine(self, original: np.ndarray, guide: np.ndarray, prompt: str) -> np.ndarray:
        if not self.api_key:
            raise RefineError("Falta la API key de OpenRouter.")

        references = [_data_url(original), _data_url(guide)]
        response = self._post("/images", {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "input_references": [
                {"type": "image_url", "image_url": {"url": url}} for url in references
            ],
        })
        if response.status_code in (400, 404, 405, 422):
            response = self._post("/chat/completions", {
                "model": self.model,
                "modalities": ["image", "text"],
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + [
                        {"type": "image_url", "image_url": {"url": url}} for url in references
                    ],
                }],
            })

        if response.status_code != 200:
            raise RefineError(f"OpenRouter respondió {response.status_code}: {_error_message(response)}")

        image = decode_image(extract_image_payload(response.json()))
        return fit_to_size(image, original.shape[1], original.shape[0])

    def _post(self, path: str, payload: dict) -> requests.Response:
        try:
            return requests.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=_headers(self.api_key),
                timeout=REQUEST_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            raise RefineError(f"No se pudo conectar con OpenRouter: {exc}") from exc


def list_image_models(api_key: str = "", base_url: str = OPENROUTER_BASE_URL) -> list[str]:
    """IDs de modelos que aceptan imagen de entrada y generan imagen."""
    try:
        response = requests.get(
            f"{base_url}/models",
            params={"output_modalities": "image"},
            headers=_headers(api_key) if api_key else None,
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    models = []
    for model in response.json().get("data", []):
        arch = model.get("architecture", {})
        if "image" in arch.get("output_modalities", []) and "image" in arch.get("input_modalities", ["image"]):
            models.append(model["id"])
    return sorted(models)


def extract_image_payload(body: dict) -> str:
    """Base64 o data URL de la primera imagen, en cualquiera de los dos formatos."""
    for item in body.get("data") or []:
        if item.get("b64_json"):
            return item["b64_json"]
        if item.get("url"):
            return item["url"]

    for choice in body.get("choices") or []:
        message = choice.get("message") or {}
        for image in message.get("images") or []:
            url = (image.get("image_url") or {}).get("url")
            if url:
                return url

    raise RefineError("El modelo no devolvió ninguna imagen. Probá con otro modelo.")


def decode_image(payload: str) -> np.ndarray:
    if payload.startswith("http"):
        try:
            content = requests.get(payload, timeout=60).content
        except requests.RequestException as exc:
            raise RefineError(f"No se pudo descargar la imagen generada: {exc}") from exc
    else:
        if payload.startswith("data:"):
            payload = payload.split(",", 1)[1]
        content = base64.b64decode(payload)

    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RefineError("La imagen generada no se pudo decodificar.")
    return image


def fit_to_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Llevar la salida al tamaño original, recortando al centro si cambia el aspecto."""
    h, w = image.shape[:2]
    target_aspect = width / height
    if abs(w / h - target_aspect) > 0.01:
        if w / h > target_aspect:
            new_w = int(round(h * target_aspect))
            x0 = (w - new_w) // 2
            image = image[:, x0:x0 + new_w]
        else:
            new_h = int(round(w / target_aspect))
            y0 = (h - new_h) // 2
            image = image[y0:y0 + new_h]
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return image


def _data_url(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "X-Title": "FacialMetrics Pro",
    }


def _error_message(response: requests.Response) -> str:
    try:
        error = response.json().get("error", {})
        return error.get("message") or response.text[:300]
    except ValueError:
        return response.text[:300]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Verificación geométrica
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    face_detected: bool
    mean_error_mm: float = float("nan")
    max_error_mm: float = float("nan")

    @property
    def passed(self) -> bool:
        return self.face_detected and self.mean_error_mm <= VERIFICATION_THRESHOLD_MM


def verify_geometry(
    detected: Optional[FaceMesh3D],
    target: FaceMesh3D,
    pixel_per_mm: float,
) -> VerificationResult:
    """
    Comparar los landmarks detectados en la salida de IA con la malla simulada.

    Se alinea primero con una similitud (escala, rotación, traslación) para no
    penalizar un reencuadre leve del modelo.
    """
    if detected is None:
        return VerificationResult(face_detected=False)

    src = detected.xy[:FACE_VERTEX_COUNT]
    dst = target.xy[:FACE_VERTEX_COUNT]
    aligned = _similarity_align(src, dst)
    errors = np.linalg.norm(aligned - dst, axis=1) / pixel_per_mm
    return VerificationResult(True, float(errors.mean()), float(errors.max()))


def _similarity_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Procrustes: transformar `src` para ajustar `dst`."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mu_s, dst - mu_d
    u, sigma, vt = np.linalg.svd(d.T @ s)
    reflection = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ reflection @ vt
    scale = (sigma * np.diag(reflection)).sum() / (s ** 2).sum()
    return (scale * (rotation @ s.T)).T + mu_d
