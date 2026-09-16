"""
simulation/viewer3d.py — Visor 3D antes/después (three.js) para Streamlit.

Renderiza la malla MediaPipe texturizada con la foto del paciente. Alrededor
del rostro se agrega un relieve de fondo (pelo, orejas, cuello y fondo de la
foto) que se aleja progresivamente hacia atrás, de modo que de frente se ve
la foto completa y al rotar el rostro conserva su volumen. Un deslizador
interpola entre la malla original y la simulada, y hay vistas rápidas de
frente, 3/4 y perfil.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np

from scipy.spatial import Delaunay

from simulation.mesh import FACE_OVAL, FACE_VERTEX_COUNT, FaceMesh3D, tessellation_triangles


THREE_VERSION = "0.160.0"
MAX_TEXTURE_SIZE = 1024

# Contornos que la teselación deja abiertos: se cierran con un abanico
EYE_LOOPS = (
    (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
)
IRIS_CENTERS = (468, 473)
MOUTH_LOOP = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)

# Anillos de fondo: (escala del óvalo facial, fracción hacia el plano de fondo)
BACKGROUND_RINGS = ((1.12, 0.3), (1.35, 0.65), (1.7, 1.0))
BACKGROUND_DEPTH = 0.2          # Plano de fondo detrás del óvalo, en alturas de rostro
BORDER_POINTS_PER_EDGE = 9

# Intensidades calibradas para que, de frente, la foto conserve su brillo original
AMBIENT_LIGHT = 2.4
KEY_LIGHT = 0.8


def build_viewer_html(
    mesh_before: FaceMesh3D,
    mesh_after: FaceMesh3D,
    texture_bgr: np.ndarray,
    height: int = 520,
    depth_scale: float = 1.0,
) -> str:
    """HTML autocontenido del visor, para `st.iframe`."""
    w, h = mesh_before.image_width, mesh_before.image_height

    # El fondo no se deforma: es igual antes y después
    background = _background_vertices(mesh_before)
    before_vertices = np.vstack([mesh_before.vertices, background])
    after_vertices = np.vstack([mesh_after.vertices, background])

    center, scale = _scene_frame(mesh_before)
    before = _normalized_vertices(before_vertices, center, scale, depth_scale)
    after = _normalized_vertices(after_vertices, center, scale, depth_scale)
    uvs = np.column_stack([before_vertices[:, 0] / w, 1.0 - before_vertices[:, 1] / h])

    face = _front_facing(np.vstack([tessellation_triangles(), _hole_triangles(mesh_before)]), before)
    ring = _front_facing(_background_triangles(mesh_before, background), before)

    data = {
        "before": np.round(before, 5).ravel().tolist(),
        "after": np.round(after, 5).ravel().tolist(),
        "uvs": np.round(uvs, 5).ravel().tolist(),
        "index": np.vstack([face, ring]).ravel().tolist(),
        "faceIndexCount": int(face.size),
        "bounds": [float(before[:, 0].min()), float(before[:, 0].max()),
                   float(before[:, 1].min()), float(before[:, 1].max())],
        "texture": _texture_data_url(texture_bgr),
    }

    return (
        _TEMPLATE
        .replace("__HEIGHT__", str(height))
        .replace("__THREE__", THREE_VERSION)
        .replace("__AMBIENT__", str(AMBIENT_LIGHT))
        .replace("__KEY__", str(KEY_LIGHT))
        .replace("__DATA__", json.dumps(data))
    )


def _scene_frame(mesh: FaceMesh3D) -> tuple[np.ndarray, float]:
    """Centro y escala de escena: el rostro mide 1 unidad de alto."""
    face = mesh.vertices[:FACE_VERTEX_COUNT]
    return face.mean(axis=0), float(np.ptp(face[:, 1])) or 1.0


def _normalized_vertices(
    vertices: np.ndarray, center: np.ndarray, scale: float, depth_scale: float
) -> np.ndarray:
    v = (vertices - center) / scale
    # three.js: y hacia arriba, z hacia la cámara
    return np.column_stack([v[:, 0], -v[:, 1], -v[:, 2] * depth_scale])


def _background_vertices(mesh: FaceMesh3D) -> np.ndarray:
    """Anillos alrededor del óvalo facial y borde de la imagen, cada vez más atrás."""
    w, h = mesh.image_width, mesh.image_height
    oval = mesh.vertices[FACE_OVAL]
    center = mesh.xy[:FACE_VERTEX_COUNT].mean(axis=0)
    face_height = float(np.ptp(mesh.xy[:FACE_VERTEX_COUNT, 1]))
    back_z = oval[:, 2].max() + BACKGROUND_DEPTH * face_height

    rings = []
    for ring_scale, toward_back in BACKGROUND_RINGS:
        xy = center + (oval[:, :2] - center) * ring_scale
        z = oval[:, 2] * (1 - toward_back) + back_z * toward_back
        rings.append(np.column_stack([xy, z]))

    t = np.linspace(0.0, 1.0, BORDER_POINTS_PER_EDGE)
    border_xy = np.vstack([
        np.column_stack([t * (w - 1), np.zeros_like(t)]),
        np.column_stack([t * (w - 1), np.full_like(t, h - 1)]),
        np.column_stack([np.zeros_like(t), t * (h - 1)]),
        np.column_stack([np.full_like(t, w - 1), t * (h - 1)]),
    ])
    rings.append(np.column_stack([border_xy, np.full(len(border_xy), back_z)]))

    points = np.vstack(rings)
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    _, unique_idx = np.unique(np.round(points[:, :2], 1), axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def _background_triangles(mesh: FaceMesh3D, background: np.ndarray) -> np.ndarray:
    """Triangulación entre el óvalo facial y el borde de la imagen (fuera del rostro)."""
    oval_xy = mesh.xy[FACE_OVAL]
    points = np.vstack([oval_xy, background[:, :2]])
    # Índices globales: óvalo → vértices de la malla; fondo → a continuación de la malla
    global_index = np.concatenate([
        np.array(FACE_OVAL),
        len(mesh.vertices) + np.arange(len(background)),
    ])

    triangles = Delaunay(points).simplices
    centroids = points[triangles].mean(axis=1)
    contour = oval_xy.astype(np.float32).reshape(-1, 1, 2)
    outside = np.array([
        cv2.pointPolygonTest(contour, (float(x), float(y)), False) < 0 for x, y in centroids
    ])
    return global_index[triangles[outside]].astype(np.int32)


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
<div id="wrap" style="width:100%;height:__HEIGHT__px;display:flex;flex-direction:column;background:#15151f;border-radius:10px;overflow:hidden;font-family:system-ui,sans-serif">
  <div id="stage" style="flex:1;min-height:0;background:radial-gradient(circle at 50% 40%, #3a3f4b 0%, #1b1d24 75%)"></div>
  <div id="controls" style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 12px;color:#e3f2fd;font-size:13px;background:#15151f">
    <span>Antes</span>
    <input id="morph" type="range" min="0" max="1" step="0.01" value="1" style="flex:1;min-width:120px">
    <span>Después</span>
    <button data-view="front">Frente</button>
    <button data-view="three">3/4</button>
    <button data-view="profile">Perfil</button>
    <label><input id="tex" type="checkbox" checked> Textura</label>
    <label><input id="bg" type="checkbox" checked> Fondo</label>
  </div>
</div>
<style>
  body { margin: 0 }
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
const stage = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(stage.clientWidth, stage.clientHeight);
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, stage.clientWidth / stage.clientHeight, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Luz casi frontal: de frente la foto conserva sus colores, al rotar se ve el volumen
scene.add(new THREE.AmbientLight(0xffffff, __AMBIENT__));
const key = new THREE.DirectionalLight(0xffffff, __KEY__);
key.position.set(0.5, 0.6, 2.0);
scene.add(key);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.Float32BufferAttribute(DATA.before, 3));
geometry.setAttribute("uv", new THREE.Float32BufferAttribute(DATA.uvs, 2));
geometry.setIndex(DATA.index);
geometry.addGroup(0, DATA.faceIndexCount, 0);
geometry.addGroup(DATA.faceIndexCount, DATA.index.length - DATA.faceIndexCount, 1);
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
const textured = new THREE.MeshStandardMaterial({ map: texture, roughness: 1, metalness: 0, side: THREE.DoubleSide });
const plain = new THREE.MeshStandardMaterial({ color: 0xd9b8a3, roughness: 0.6, side: THREE.DoubleSide });
const texturedBg = textured.clone();
const plainBg = new THREE.MeshStandardMaterial({ color: 0x8a8f99, roughness: 1, side: THREE.DoubleSide });
const mesh = new THREE.Mesh(geometry, [textured, texturedBg]);
scene.add(mesh);

// Encuadre de la foto completa
const [minX, maxX, minY, maxY] = DATA.bounds;
const target = new THREE.Vector3((minX + maxX) / 2, (minY + maxY) / 2, 0);
function fitDistance() {
  const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  const hx = (maxX - minX) / 2, hy = (maxY - minY) / 2;
  return 1.04 * Math.max(hy, hx / camera.aspect) / Math.tan(halfFov);
}
const views = {
  front: [0, 0, 1],
  three: [0.62, 0.08, 0.78],
  profile: [1, 0.03, 0],
};
function setView(name) {
  const d = fitDistance();
  const [x, y, z] = views[name];
  camera.position.set(target.x + x * d, target.y + y * d, target.z + z * d);
  controls.target.copy(target);
  controls.update();
}
setView("front");

document.getElementById("morph").addEventListener("input", (e) => {
  morph(parseFloat(e.target.value));
});
const texInput = document.getElementById("tex");
const bgInput = document.getElementById("bg");
function updateMaterials() {
  mesh.material = texInput.checked ? [textured, texturedBg] : [plain, plainBg];
  texturedBg.visible = plainBg.visible = bgInput.checked;
}
texInput.addEventListener("change", updateMaterials);
bgInput.addEventListener("change", updateMaterials);
document.querySelectorAll("#controls button").forEach((b) =>
  b.addEventListener("click", () => setView(b.dataset.view))
);
window.addEventListener("resize", () => {
  camera.aspect = stage.clientWidth / stage.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(stage.clientWidth, stage.clientHeight);
});

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
</script>
"""
