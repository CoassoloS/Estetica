"""
test_model3d.py — Tests del cliente de modelos 3D en fal.ai (API mockeada).

Verifica:
    - Cola de fal.ai: envío, consulta de estado con espera creciente y resultado
    - Endpoints y payloads de Rodin V2.5, Hunyuan3D 3.1 Pro y TRELLIS
    - Manejo de 429, errores del modelo y salidas sin archivo 3D
    - Caché en disco: no se vuelve a llamar a la API con las mismas fotos
"""

import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.model3d import FalClient, Model3DError
from simulation.viewer_glb import build_glb_viewer_html

GLB_BYTES = b"glTF\x02\x00\x00\x00fake-model"
STATUS_URL = "https://queue.fal.run/fal-ai/trellis/requests/req-1/status"
RESPONSE_URL = "https://queue.fal.run/fal-ai/trellis/requests/req-1"


def response(status: int, body=None, headers=None, content: bytes = b"") -> MagicMock:
    mock = MagicMock(status_code=status, headers=headers or {}, content=content, text=str(body))
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()
    return mock


def image(value: int = 120) -> np.ndarray:
    return np.full((40, 30, 3), value, dtype=np.uint8)


SUBMITTED = response(200, {
    "request_id": "req-1", "status_url": STATUS_URL, "response_url": RESPONSE_URL,
    "queue_position": 0,
})
COMPLETED = response(200, {"status": "COMPLETED"})


def trellis_output():
    return response(200, {"model_mesh": {"url": "https://fal.media/files/model.glb"}, "timings": {}})


def make_client(tmp_path, model_key="trellis", **kwargs):
    sleeps: list[float] = []
    client = FalClient(api_key="key", model_key=model_key, cache_dir=tmp_path,
                       sleep=sleeps.append, clock=lambda: 0.0, **kwargs)
    return client, sleeps


class TestGenerate:

    def test_trellis_multi_view_flow(self, tmp_path):
        client, sleeps = make_client(tmp_path)
        progress = []
        calls = [
            SUBMITTED,
            response(200, {"status": "IN_QUEUE", "queue_position": 2}),
            response(200, {"status": "IN_PROGRESS", "logs": []}),
            COMPLETED,
            trellis_output(),
        ]
        with patch("simulation.model3d.requests.request", side_effect=calls) as req, \
             patch("simulation.model3d.requests.get",
                   return_value=response(200, content=GLB_BYTES)) as get:
            result = client.generate([image(), image(90)],
                                     on_progress=lambda s, q: progress.append((s, q)))

        assert result.glb == GLB_BYTES and not result.from_cache

        method, url = req.call_args_list[0].args
        assert (method, url) == ("POST", "https://queue.fal.run/fal-ai/trellis/multi")
        assert req.call_args_list[0].kwargs["headers"]["Authorization"] == "Key key"
        payload = req.call_args_list[0].kwargs["json"]
        assert len(payload["image_urls"]) == 2
        assert payload["image_urls"][0].startswith("data:image/jpeg;base64,")

        assert req.call_args_list[1].args == ("GET", STATUS_URL)
        assert req.call_args_list[4].args == ("GET", RESPONSE_URL)
        assert get.call_args.args[0] == "https://fal.media/files/model.glb"
        assert progress == [("Waiting", 2), ("Generating", None)]
        assert sleeps == [2.0, 3.0, 4.5]

    def test_trellis_single_view(self, tmp_path):
        client, _ = make_client(tmp_path)
        with patch("simulation.model3d.requests.request",
                   side_effect=[SUBMITTED, COMPLETED, trellis_output()]) as req, \
             patch("simulation.model3d.requests.get", return_value=response(200, content=GLB_BYTES)):
            client.generate([image()])
        assert req.call_args_list[0].args[1].endswith("/fal-ai/trellis")
        assert "image_url" in req.call_args_list[0].kwargs["json"]

    def test_rodin_sends_all_views_as_glb(self, tmp_path):
        client, _ = make_client(tmp_path, model_key="rodin")
        output = response(200, {"model_mesh": {"url": "https://fal.media/files/r.glb"}, "textures": []})
        with patch("simulation.model3d.requests.request",
                   side_effect=[SUBMITTED, COMPLETED, output]) as req, \
             patch("simulation.model3d.requests.get", return_value=response(200, content=GLB_BYTES)) as get:
            client.generate([image(), image(90)])

        assert req.call_args_list[0].args[1].endswith("/fal-ai/hyper3d/rodin/v2.5")
        payload = req.call_args_list[0].kwargs["json"]
        assert len(payload["image_urls"]) == 2
        assert payload["geometry_file_format"] == "glb" and payload["material"] == "PBR"
        assert get.call_args.args[0] == "https://fal.media/files/r.glb"

    def test_hunyuan_pro_uses_only_frontal(self, tmp_path):
        client, _ = make_client(tmp_path, model_key="hunyuan_pro")
        output = response(200, {"model_glb": {"url": "https://fal.media/files/h.glb"}})
        with patch("simulation.model3d.requests.request",
                   side_effect=[SUBMITTED, COMPLETED, output]) as req, \
             patch("simulation.model3d.requests.get", return_value=response(200, content=GLB_BYTES)) as get:
            client.generate([image(), image(90)])

        assert req.call_args_list[0].args[1].endswith("/fal-ai/hunyuan-3d/v3.1/pro/image-to-3d")
        assert set(req.call_args_list[0].kwargs["json"]) == {"input_image_url", "enable_pbr"}
        assert get.call_args.args[0] == "https://fal.media/files/h.glb"
        # La foto lateral no cambia el caché: el modelo solo usa la frontal
        assert client.is_cached([image()]) and client.is_cached([image(), image(10)])

    def test_cache_avoids_second_call(self, tmp_path):
        client, _ = make_client(tmp_path)
        with patch("simulation.model3d.requests.request",
                   side_effect=[SUBMITTED, COMPLETED, trellis_output()]), \
             patch("simulation.model3d.requests.get", return_value=response(200, content=GLB_BYTES)):
            client.generate([image()])

        assert client.is_cached([image()]) and not client.is_cached([image(10)])
        keyless = FalClient(api_key="", model_key="trellis", cache_dir=tmp_path)
        with patch("simulation.model3d.requests.request") as req:
            result = keyless.generate([image()])
        req.assert_not_called()
        assert result.from_cache and result.glb == GLB_BYTES

    def test_model_is_part_of_cache_key(self, tmp_path):
        rodin = FalClient(api_key="k", cache_dir=tmp_path)
        trellis = FalClient(api_key="k", model_key="trellis", cache_dir=tmp_path)
        assert rodin.model_key == "rodin"
        assert rodin._cache_path([b"a"]) != trellis._cache_path([b"a"])


class TestErrors:

    def test_rate_limit_retries(self, tmp_path):
        client, sleeps = make_client(tmp_path)
        with patch("simulation.model3d.requests.request",
                   side_effect=[response(429, {}, headers={"Retry-After": "7"}), SUBMITTED,
                                COMPLETED, trellis_output()]), \
             patch("simulation.model3d.requests.get", return_value=response(200, content=GLB_BYTES)):
            client.generate([image()])
        assert sleeps[0] == 7.0

    def test_model_error_raises(self, tmp_path):
        client, _ = make_client(tmp_path)
        failed = response(200, {"status": "COMPLETED", "error": "Image processing failed"})
        with patch("simulation.model3d.requests.request", side_effect=[SUBMITTED, failed]):
            with pytest.raises(Model3DError, match="Image processing failed"):
                client.generate([image()])
        assert not client.is_cached([image()])

    def test_http_error_raises(self, tmp_path):
        client, _ = make_client(tmp_path)
        with patch("simulation.model3d.requests.request",
                   return_value=response(401, {"detail": "Invalid key"})):
            with pytest.raises(Model3DError, match="Invalid key"):
                client.generate([image()])

    def test_missing_output_file_raises(self, tmp_path):
        client, _ = make_client(tmp_path)
        with patch("simulation.model3d.requests.request",
                   side_effect=[SUBMITTED, COMPLETED, response(200, {"timings": {}})]):
            with pytest.raises(Model3DError, match="timings"):
                client.generate([image()])

    def test_missing_key_raises(self, tmp_path):
        with pytest.raises(Model3DError, match="API key"):
            FalClient(api_key="", cache_dir=tmp_path).generate([image()])

    def test_requires_images(self, tmp_path):
        with pytest.raises(Model3DError):
            FalClient(api_key="k", cache_dir=tmp_path).generate([])


class TestGlbViewer:

    def test_embeds_models(self):
        html = build_glb_viewer_html(GLB_BYTES, None)
        assert '{"before": "' in html and '"after": "' not in html
        assert "__MODELS__" not in html and "GLTFLoader" in html

    def test_requires_a_model(self):
        with pytest.raises(ValueError):
            build_glb_viewer_html(None, None)
