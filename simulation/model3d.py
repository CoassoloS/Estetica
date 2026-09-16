"""
simulation/model3d.py — Cabeza 3D completa con modelos generativos en fal.ai.

Genera un modelo 3D completo (nuca, pelo, orejas) a partir de fotos, con
modelos servidos por fal.ai con pago por uso:

    - Rodin V2.5 (Hyper3D): máxima calidad y realismo; acepta varias vistas.
    - Hunyuan3D 3.1 Pro (Tencent): alta calidad; usa la foto frontal.
    - TRELLIS (Microsoft): rápido y muy barato; acepta varias vistas.

La geometría es generativa: sirve para mostrar al paciente, no para medir.

Flujo de la cola de fal.ai (asíncrona):
    1. POST https://queue.fal.run/{endpoint}   → request_id, status_url, response_url
    2. GET status_url                          → IN_QUEUE | IN_PROGRESS | COMPLETED
    3. GET response_url                        → salida del modelo con la URL del .glb

Los modelos se guardan en disco por huella de las fotos y el modelo, para no
pagar dos veces la misma generación y poder dejarlos listos antes de una demo.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import requests


FAL_QUEUE_URL = "https://queue.fal.run"
MAX_INPUT_SIZE = 1024           # Lado mayor de las fotos enviadas (px)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "model3d_cache"

POLL_INITIAL_S = 2.0
POLL_MAX_S = 10.0
DEADLINE_S = 20 * 60


class Model3DError(RuntimeError):
    """Error del proveedor 3D, con mensaje apto para mostrar al usuario."""


@dataclass(frozen=True)
class FalModel:
    key: str
    label: str
    endpoint: str
    max_images: int                     # Fotos que usa (la primera es la frontal)
    output_field: str                   # Campo de la salida con el archivo .glb
    multi_endpoint: Optional[str] = None  # Endpoint alternativo cuando hay varias fotos

    def request(self, image_urls: list[str]) -> tuple[str, dict]:
        """Endpoint y payload para las fotos dadas (data URLs)."""
        urls = image_urls[:self.max_images]
        if self.key == "rodin":
            return self.endpoint, {
                "image_urls": urls,
                "tier": "Gen-2.5-High",
                "material": "PBR",
                "geometry_file_format": "glb",
            }
        if self.key == "hunyuan_pro":
            return self.endpoint, {"input_image_url": urls[0], "enable_pbr": True}
        if len(urls) > 1 and self.multi_endpoint:
            return self.multi_endpoint, {"image_urls": urls}
        return self.endpoint, {"image_url": urls[0]}

    def images_used(self, count: int) -> int:
        return min(count, self.max_images)


FAL_MODELS: dict[str, FalModel] = {
    "rodin": FalModel(
        key="rodin",
        label="Rodin V2.5 — máxima calidad (≈ US$ 0,40 por modelo)",
        endpoint="fal-ai/hyper3d/rodin/v2.5",
        max_images=5,
        output_field="model_mesh",
    ),
    "hunyuan_pro": FalModel(
        key="hunyuan_pro",
        label="Hunyuan3D 3.1 Pro — alta calidad (≈ US$ 0,38 por modelo)",
        endpoint="fal-ai/hunyuan-3d/v3.1/pro/image-to-3d",
        max_images=1,
        output_field="model_glb",
    ),
    "trellis": FalModel(
        key="trellis",
        label="TRELLIS — rápido y económico (≈ US$ 0,02 por modelo)",
        endpoint="fal-ai/trellis",
        max_images=5,
        output_field="model_mesh",
        multi_endpoint="fal-ai/trellis/multi",
    ),
}
DEFAULT_MODEL_KEY = "rodin"


@dataclass
class Model3DResult:
    glb: bytes
    from_cache: bool = False


ProgressCallback = Callable[[str, Optional[int]], None]


@dataclass
class FalClient:
    api_key: str
    model_key: str = DEFAULT_MODEL_KEY
    base_url: str = FAL_QUEUE_URL
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    @property
    def model(self) -> FalModel:
        return FAL_MODELS[self.model_key]

    # ── Flujo completo ───────────────────────────────────────────────────

    def generate(
        self,
        images: list[np.ndarray],
        on_progress: Optional[ProgressCallback] = None,
    ) -> Model3DResult:
        """
        Generar (o recuperar del caché) el modelo GLB para las fotos dadas.

        Args:
            images: Imágenes BGR; la primera es la frontal. Los modelos de una
                sola vista usan solo la primera.
            on_progress: Callback (estado "Waiting" | "Generating", posición en la cola).
        """
        if not images:
            raise Model3DError("Se necesita al menos una foto.")

        encoded = self._encode(images)
        cache_path = self._cache_path(encoded)
        if cache_path is not None and cache_path.exists():
            return Model3DResult(cache_path.read_bytes(), from_cache=True)

        if not self.api_key:
            raise Model3DError("Falta la API key de fal.ai (FAL_KEY).")

        endpoint, payload = self.model.request(
            ["data:image/jpeg;base64," + base64.b64encode(data).decode("ascii") for data in encoded]
        )
        submitted = self._request("POST", f"{self.base_url}/{endpoint}", json=payload)
        try:
            status_url, response_url = submitted["status_url"], submitted["response_url"]
        except (KeyError, TypeError) as exc:
            raise Model3DError(f"Respuesta inesperada de fal.ai al enviar: {submitted}") from exc

        self._wait(status_url, on_progress)
        output = self._request("GET", response_url)
        glb = self._download_glb(output)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(glb)
        return Model3DResult(glb)

    def is_cached(self, images: list[np.ndarray]) -> bool:
        path = self._cache_path(self._encode(images))
        return path is not None and path.exists()

    # ── Pasos ────────────────────────────────────────────────────────────

    def _wait(self, status_url: str, on_progress: Optional[ProgressCallback]) -> None:
        deadline = self.clock() + DEADLINE_S
        delay = POLL_INITIAL_S
        while True:
            self.sleep(delay)
            body = self._request("GET", status_url)
            if body.get("error"):
                raise Model3DError(f"fal.ai no pudo generar el modelo: {body['error']}")

            status = body.get("status")
            if status == "COMPLETED":
                return
            if on_progress is not None:
                state = "Generating" if status == "IN_PROGRESS" else "Waiting"
                on_progress(state, body.get("queue_position"))
            if self.clock() + delay > deadline:
                raise Model3DError("La generación tardó más de 20 minutos. Reintentá más tarde.")
            delay = min(delay * 1.5, POLL_MAX_S)

    def _download_glb(self, output: dict) -> bytes:
        file_info = output.get(self.model.output_field) or {}
        url = file_info.get("url") if isinstance(file_info, dict) else None
        if not url:
            fields = ", ".join(output) or "ninguno"
            raise Model3DError(f"fal.ai no devolvió el modelo 3D (campos: {fields}).")
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise Model3DError(f"No se pudo descargar el modelo 3D: {exc}") from exc
        return response.content

    def _request(self, method: str, url: str, **kwargs) -> dict:
        for _ in range(5):
            try:
                response = requests.request(
                    method, url,
                    headers={"Authorization": f"Key {self.api_key}"},
                    timeout=120,
                    **kwargs,
                )
            except requests.RequestException as exc:
                raise Model3DError(f"No se pudo conectar con fal.ai: {exc}") from exc

            if response.status_code == 429:
                self.sleep(_retry_after(response))
                continue
            if response.status_code not in (200, 201, 202):
                raise Model3DError(f"fal.ai respondió {response.status_code}: {_error_message(response)}")
            try:
                return response.json()
            except ValueError as exc:
                raise Model3DError("fal.ai devolvió una respuesta no válida.") from exc

        raise Model3DError("fal.ai limitó las solicitudes (429). Esperá un momento y reintentá.")

    # ── Utilidades ───────────────────────────────────────────────────────

    def _encode(self, images: list[np.ndarray]) -> list[bytes]:
        return [_encode_jpeg(img) for img in images[:self.model.images_used(len(images))]]

    def _cache_path(self, encoded_images: list[bytes]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(self.model_key.encode())
        for data in encoded_images:
            digest.update(hashlib.sha256(data).digest())
        return Path(self.cache_dir) / f"{digest.hexdigest()[:32]}.glb"


def _encode_jpeg(image: np.ndarray) -> bytes:
    h, w = image.shape[:2]
    scale = min(1.0, MAX_INPUT_SIZE / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise Model3DError("No se pudo codificar una imagen.")
    return buffer.tobytes()


def _retry_after(response: requests.Response) -> float:
    try:
        return max(1.0, float(response.headers.get("Retry-After", POLL_INITIAL_S)))
    except ValueError:
        return POLL_INITIAL_S


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
        return str(body.get("detail") or body.get("error") or body)[:300]
    except ValueError:
        return response.text[:300]
