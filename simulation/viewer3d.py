"""
simulation/viewer3d.py — Visor 3D antes/después (three.js) para Streamlit.

Renderiza la malla MediaPipe texturizada con la foto del paciente. Un
deslizador interpola entre la malla original y la simulada, y hay
vistas rápidas de frente, 3/4 y perfil, donde se aprecian los cambios de
proyección (mentón, punta nasal, pómulos) que no se ven en la foto frontal.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np

from simulation.mesh import FaceMesh3D, tessellation_triangles


THREE_VERSION = "0.160.0"
MAX_TEXTURE_SIZE = 1024

# Contornos que la teselación deja abiertos: se cierran con un abanico
EYE_LOOPS = (
    (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
)
IRIS_CENTERS = (468, 473)
MOUTH_LOOP = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)


def build_viewer_html(
    mesh_before: FaceMesh3D,
    mesh_after: FaceMesh3D,
    texture_bgr: np.ndarray,
    height: int = 520,
    depth_scale: float = 1.0,
) -> str:
    """HTML autocontenido del visor, para `st.iframe`."""
    triangles = np.vstack([tessellation_triangles(), _hole_triangles(mesh_before)])
    w, h = mesh_before.image_width, mesh_before.image_height

    before = _normalized_vertices(mesh_before, mesh_before, depth_scale)
    after = _normalized_vertices(mesh_after, mesh_before, depth_scale)
    uvs = np.column_stack([
        mesh_before.vertices[:, 0] / w,
        1.0 - mesh_before.vertices[:, 1] / h,
    ])
    triangles = _front_facing(triangles, before)

    data = {
        "before": np.round(before, 5).ravel().tolist(),
        "after": np.round(after, 5).ravel().tolist(),
        "uvs": np.round(uvs, 5).ravel().tolist(),
        "index": triangles.ravel().tolist(),
        "texture": _texture_data_url(texture_bgr),
    }

    return (
        _TEMPLATE
        .replace("__HEIGHT__", str(height))
        .replace("__THREE__", THREE_VERSION)
        .replace("__DATA__", json.dumps(data))
    )


def _normalized_vertices(mesh: FaceMesh3D, reference: FaceMesh3D, depth_scale: float) -> np.ndarray:
    """Centrar y escalar a unidades de escena usando la malla de referencia."""
    ref = reference.vertices
    center = ref.mean(axis=0)
    scale = float(np.ptp(ref[:, 1])) or 1.0
    v = (mesh.vertices - center) / scale
    # three.js: y hacia arriba, z hacia la cámara
    return np.column_stack([v[:, 0], -v[:, 1], -v[:, 2] * depth_scale])


def _hole_triangles(mesh: FaceMesh3D) -> np.ndarray:
    """Triángulos que cierran ojos (desde el centro del iris) y boca."""
    fans = []
    for loop in EYE_LOOPS:
        loop_center = mesh.xy[list(loop)].mean(axis=0)
        center = min(IRIS_CENTERS, key=lambda i: np.linalg.norm(mesh.xy[i] - loop_center))
        fans.append((center, loop))
    fans.append((MOUTH_LOOP[0], MOUTH_LOOP[1:]))

    triangles = []
    for center, loop in fans:
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if center not in (a, b):
                triangles.append((center, a, b))
    return np.array(triangles, dtype=np.int32)


def _front_facing(triangles: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Orientar los triángulos para que su normal apunte hacia la cámara (+z)."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    normal_z = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    oriented = triangles.copy()
    flip = normal_z < 0
    oriented[flip] = oriented[flip][:, [0, 2, 1]]
    return oriented


def _texture_data_url(image_bgr: np.ndarray) -> str:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, MAX_TEXTURE_SIZE / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


_TEMPLATE = """
<meta charset="utf-8">
<div id="wrap" style="position:relative;width:100%;height:__HEIGHT__px;background:#15151f;border-radius:10px;overflow:hidden;font-family:system-ui,sans-serif">
  <div id="controls" style="position:absolute;left:12px;right:12px;bottom:10px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;color:#e3f2fd;font-size:13px;z-index:2">
    <span>Antes</span>
    <input id="morph" type="range" min="0" max="1" step="0.01" value="1" style="flex:1;min-width:120px">
    <span>Después</span>
    <button data-view="front">Frente</button>
    <button data-view="three">3/4</button>
    <button data-view="profile">Perfil</button>
    <label><input id="tex" type="checkbox" checked> Textura</label>
  </div>
</div>
<style>
  #controls button { background:#1565c0;color:#fff;border:0;border-radius:6px;padding:4px 10px;cursor:pointer }
  #controls button:hover { background:#1e88e5 }
</style>
<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@__THREE__/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@__THREE__/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DATA = __DATA__;
const wrap = document.getElementById("wrap");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(wrap.clientWidth, wrap.clientHeight);
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15151f);
const camera = new THREE.PerspectiveCamera(30, wrap.clientWidth / wrap.clientHeight, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(1.2, 1.0, 2.0);
scene.add(key);
const rim = new THREE.DirectionalLight(0xffffff, 0.5);
rim.position.set(-2.0, 0.5, -0.5);
scene.add(rim);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.Float32BufferAttribute(DATA.before, 3));
geometry.setAttribute("uv", new THREE.Float32BufferAttribute(DATA.uvs, 2));
geometry.setIndex(DATA.index);
const positions = geometry.getAttribute("position");
function morph(t) {
  for (let i = 0; i < DATA.before.length; i++) {
    positions.array[i] = DATA.before[i] + t * (DATA.after[i] - DATA.before[i]);
  }
  positions.needsUpdate = true;
  geometry.computeVertexNormals();   // el sombreado refleja los cambios de volumen
}
morph(1);

const texture = new THREE.TextureLoader().load(DATA.texture);
texture.colorSpace = THREE.SRGBColorSpace;
const textured = new THREE.MeshStandardMaterial({ map: texture, roughness: 0.75, side: THREE.DoubleSide });
const plain = new THREE.MeshStandardMaterial({ color: 0xd9b8a3, roughness: 0.6, side: THREE.DoubleSide });
const mesh = new THREE.Mesh(geometry, textured);
scene.add(mesh);

const views = {
  front: [0, 0, 3.2],
  three: [2.1, 0.2, 2.4],
  profile: [3.2, 0.1, 0],
};
function setView(name) {
  const [x, y, z] = views[name];
  camera.position.set(x, y, z);
  controls.target.set(0, 0, 0);
  controls.update();
}
setView("front");

document.getElementById("morph").addEventListener("input", (e) => {
  morph(parseFloat(e.target.value));
});
document.getElementById("tex").addEventListener("change", (e) => {
  mesh.material = e.target.checked ? textured : plain;
});
document.querySelectorAll("#controls button").forEach((b) =>
  b.addEventListener("click", () => setView(b.dataset.view))
);
window.addEventListener("resize", () => {
  camera.aspect = wrap.clientWidth / wrap.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
});

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
</script>
"""
