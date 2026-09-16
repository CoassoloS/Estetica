"""
test_ai_refine.py — Tests del refinado con IA (OpenRouter mockeado).

Verifica:
    - Lectura de imágenes en ambos formatos de respuesta de OpenRouter
    - Fallback de /images a chat completions y manejo de errores
    - Ajuste de tamaño de la salida y verificación geométrica
"""

import base64
import sys
import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import FaceMesh3D
from simulation.ai_refine import (
    OpenRouterBackend,
    RefineError,
    build_prompt,
    extract_image_payload,
    fit_to_size,
    verify_geometry,
)
from tests.sim_fixtures import make_synthetic_vertices


def png_b64(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", image)[1]).decode()


def response(status: int, body: dict) -> MagicMock:
    mock = MagicMock(status_code=status, text=str(body))
    mock.json.return_value = body
    return mock


class TestPayload:

    def test_images_api_format(self):
        assert extract_image_payload({"data": [{"b64_json": "abc"}]}) == "abc"

    def test_chat_format(self):
        body = {"choices": [{"message": {"images": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}
        ]}}]}
        assert extract_image_payload(body) == "data:image/png;base64,xyz"

    def test_no_image_raises(self):
        with pytest.raises(RefineError):
            extract_image_payload({"choices": [{"message": {"content": "no puedo"}}]})


class TestBackend:

    def setup_method(self):
        self.original = np.full((100, 80, 3), 120, dtype=np.uint8)
        self.generated = np.full((200, 160, 3), 200, dtype=np.uint8)

    def test_fallback_to_chat_completions(self):
        chat_body = {"choices": [{"message": {"images": [
            {"image_url": {"url": "data:image/png;base64," + png_b64(self.generated)}}
        ]}}]}
        with patch("simulation.ai_refine.requests.post",
                   side_effect=[response(404, {}), response(200, chat_body)]) as post:
            result = OpenRouterBackend("key", "m").refine(self.original, self.original, "p")

        assert post.call_args_list[0].args[0].endswith("/images")
        assert post.call_args_list[1].args[0].endswith("/chat/completions")
        assert post.call_args_list[1].kwargs["json"]["modalities"] == ["image", "text"]
        assert post.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer key"
        assert result.shape == self.original.shape
        assert result.mean() == pytest.approx(200, abs=1)

    def test_images_api_success(self):
        body = {"data": [{"b64_json": png_b64(self.generated)}]}
        with patch("simulation.ai_refine.requests.post", return_value=response(200, body)) as post:
            OpenRouterBackend("key", "m").refine(self.original, self.original, "p")
        refs = post.call_args.kwargs["json"]["input_references"]
        assert len(refs) == 2 and refs[0]["image_url"]["url"].startswith("data:image/jpeg")

    def test_http_error_raises(self):
        with patch("simulation.ai_refine.requests.post",
                   return_value=response(401, {"error": {"message": "invalid key"}})):
            with pytest.raises(RefineError, match="invalid key"):
                OpenRouterBackend("bad", "m").refine(self.original, self.original, "p")

    def test_missing_key_raises(self):
        with pytest.raises(RefineError):
            OpenRouterBackend("", "m").refine(self.original, self.original, "p")


class TestHelpers:

    def test_fit_to_size_crops_aspect(self):
        wide = np.zeros((100, 300, 3), dtype=np.uint8)
        wide[:, 100:200] = 255
        fitted = fit_to_size(wide, 50, 50)
        assert fitted.shape == (50, 50, 3)
        assert fitted.mean() == pytest.approx(255, abs=5)

    def test_prompt_lists_changes(self):
        prompt = build_prompt(["fuller upper lip"])
        assert "- fuller upper lip" in prompt
        assert "identity" in prompt


class TestVerification:

    def make_mesh(self, vertices):
        return FaceMesh3D(vertices=vertices, image_width=400, image_height=500)

    def test_similarity_transform_passes(self):
        target = make_synthetic_vertices()
        moved = target.copy()
        moved[:, :2] = moved[:, :2] * 1.1 + 15
        result = verify_geometry(self.make_mesh(moved), self.make_mesh(target), 3.0)
        assert result.passed
        assert result.mean_error_mm == pytest.approx(0, abs=1e-6)

    def test_distorted_geometry_fails(self):
        target = make_synthetic_vertices()
        moved = target.copy()
        moved[:, 0] += np.random.default_rng(0).normal(0, 15, len(moved))
        result = verify_geometry(self.make_mesh(moved), self.make_mesh(target), 3.0)
        assert not result.passed

    def test_no_face(self):
        result = verify_geometry(None, self.make_mesh(make_synthetic_vertices()), 3.0)
        assert not result.face_detected and not result.passed
