"""
simulation/viewer3d.py — Visor 3D antes/después (three.js) para Streamlit.

Renderiza la malla MediaPipe texturizada con la foto del paciente, como un
busto en relieve: alrededor del rostro, pelo, orejas y cuello se curvan
hacia atrás y se desvanecen en el borde. Como una sola foto frontal no tiene
información de los laterales de la cabeza, la rotación se limita a ±40° para
que el volumen se vea creíble. Un deslizador interpola entre la malla
original y la simulada.

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

# Anillos del busto: (escala del óvalo facial, curvatura hacia atrás 0-1, opacidad)
BACKGROUND_RINGS = (
    (1.10, 0.15, 1.0),
    (1.25, 0.40, 1.0),
    (1.45, 0.70, 0.75),
    (1.70, 1.00, 0.0),
)
BACKGROUND_DEPTH = 0.55         # Profundidad del borde exterior detrás del óvalo, en alturas de rostro
BACKGROUND_WRAP = 0.12          # Cuánto se cierra hacia adentro el borde exterior al curvarse

# Recorte del fondo de la foto (fondos clínicos uniformes)
BACKDROP_BORDER_PX = 6
BACKDROP_MAX_STD = 18.0         # Si el borde no es uniforme, no se recorta
BACKDROP_DISTANCE = (10.0, 26.0)  # Distancia Lab: transparente → opaco

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

    # El relieve no se deforma: es igual antes y después
    background, background_uv, background_alpha = _background_vertices(mesh_before)
    before_vertices = np.vstack([mesh_before.vertices, background])
    after_vertices = np.vstack([mesh_after.vertices, background])

    center, scale = _scene_frame(mesh_before)
    before = _normalized_vertices(before_vertices, center, scale, depth_scale)
    after = _normalized_vertices(after_vertices, center, scale, depth_scale)
    uv_xy = np.vstack([mesh_before.xy, background_uv])
    uvs = np.column_stack([uv_xy[:, 0] / w, 1.0 - uv_xy[:, 1] / h])
    alpha = np.concatenate([np.ones(len(mesh_before.vertices)), background_alpha])

    face = _front_facing(np.vstack([tessellation_triangles(), _hole_triangles(mesh_before)]), before)
    ring = _front_facing(_background_triangles(mesh_before, background_uv), before)

    data = {
        "before": np.round(before, 5).ravel().tolist(),
        "after": np.round(after, 5).ravel().tolist(),
        "uvs": np.round(uvs, 5).ravel().tolist(),
        "alpha": np.round(alpha, 3).tolist(),
        "index": np.vstack([face, ring]).ravel().tolist(),
        "faceIndexCount": int(face.size),
        "bounds": _visible_bounds(before, uv_xy, w, h),
        "texture": _texture_data_url(texture_bgr, mesh_before),
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


def _background_vertices(mesh: FaceMesh3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Anillos del busto alrededor del óvalo facial.

    Returns:
        (vértices 3D, posición en la foto para la textura, opacidad) por vértice.
        Los anillos se curvan hacia atrás y hacia adentro, pero conservan la
        coordenada de textura de su posición en la foto.
    """
    oval = mesh.vertices[FACE_OVAL]
    center = mesh.xy[:FACE_VERTEX_COUNT].mean(axis=0)
    face_height = float(np.ptp(mesh.xy[:FACE_VERTEX_COUNT, 1]))
    back_z = oval[:, 2].max() + BACKGROUND_DEPTH * face_height

    vertices, uv_xy, alpha = [], [], []
    for ring_scale, curve, opacity in BACKGROUND_RINGS:
        # Sin recortar al borde: fuera de la foto la textura queda transparente
        xy = center + (oval[:, :2] - center) * ring_scale
        # Curva suave (cuadrática): cerca del rostro casi no se aleja
        depth = curve ** 2
        geometry_xy = center + (xy - center) * (1 - BACKGROUND_WRAP * curve)
        z = oval[:, 2] * (1 - depth) + back_z * depth
        vertices.append(np.column_stack([geometry_xy, z]))
        uv_xy.append(xy)
        alpha.append(np.full(len(xy), opacity))

    vertices, uv_xy, alpha = np.vstack(vertices), np.vstack(uv_xy), np.concatenate(alpha)
    _, unique_idx = np.unique(np.round(uv_xy, 1), axis=0, return_index=True)
    keep = np.sort(unique_idx)
    return vertices[keep], uv_xy[keep], alpha[keep]


def _background_triangles(mesh: FaceMesh3D, background_uv: np.ndarray) -> np.ndarray:
    """Triangulación de los anillos del busto (fuera del rostro), en el plano de la foto."""
    oval_xy = mesh.xy[FACE_OVAL]
    points = np.vstack([oval_xy, background_uv])
    # Índices globales: óvalo → vértices de la malla; fondo → a continuación de la malla
    global_index = np.concatenate([
        np.array(FACE_OVAL),
        len(mesh.vertices) + np.arange(len(background_uv)),
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


def _visible_bounds(vertices: np.ndarray, uv_xy: np.ndarray, w: int, h: int) -> list[float]:
    """Límites en escena de los vértices que caen dentro de la foto."""
    inside = (uv_xy[:, 0] >= 0) & (uv_xy[:, 0] <= w) & (uv_xy[:, 1] >= 0) & (uv_xy[:, 1] <= h)
    v = vertices[inside]
    return [float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 1].min()), float(v[:, 1].max())]


def _texture_data_url(image_bgr: np.ndarray, mesh: FaceMesh3D) -> str:
    """Textura PNG con el fondo de la foto transparente."""
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = backdrop_alpha(image_bgr, mesh)

    h, w = rgba.shape[:2]
    scale = min(1.0, MAX_TEXTURE_SIZE / max(h, w))
    if scale < 1.0:
        rgba = cv2.resize(rgba, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    # Borde transparente de 1 px: fuera de la foto la textura se estira transparente
    rgba[0, :, 3] = rgba[-1, :, 3] = rgba[:, 0, 3] = rgba[:, -1, 3] = 0
    _, buffer = cv2.imencode(".png", rgba)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def backdrop_alpha(image_bgr: np.ndarray, mesh: FaceMesh3D) -> np.ndarray:
    """
    Opacidad (0-255) que vuelve transparente el fondo uniforme de la foto.

    Solo se quitan regiones del color del fondo conectadas con el borde de la
    imagen, y nunca el interior del rostro (dientes y escleras son claros).
    """
    h, w = image_bgr.shape[:2]
    opaque = np.full((h, w), 255, dtype=np.uint8)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = BACKDROP_BORDER_PX
    border = np.vstack([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ])
    backdrop = np.median(border, axis=0)
    close = np.linalg.norm(border - backdrop, axis=1) < BACKDROP_DISTANCE[1]
    if border[close].std(axis=0).max() > BACKDROP_MAX_STD or close.mean() < 0.5:
        return opaque

    distance = np.linalg.norm(lab - backdrop, axis=2)
    low, high = BACKDROP_DISTANCE
    alpha = np.clip((distance - low) / (high - low), 0.0, 1.0)

    # Solo fondo conectado al borde de la imagen
    candidate = (alpha < 0.5).astype(np.uint8)
    _, labels = cv2.connectedComponents(candidate, connectivity=4)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    edge_labels = edge_labels[edge_labels != 0]
    backdrop_region = np.isin(labels, edge_labels)
    backdrop_region = cv2.dilate(backdrop_region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    alpha = np.where(backdrop_region, alpha, 1.0)

    # El rostro siempre opaco
    face = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face, [np.round(mesh.xy[FACE_OVAL]).astype(np.int32)], 1)
    alpha[face > 0] = 1.0

    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.2)
    return np.clip(alpha * 255, 0, 255).astype(np.uint8)


_TEMPLATE = """
<meta charset="utf-8">
<div id="wrap" style="width:100%;height:__HEIGHT__px;display:flex;flex-direction:column;background:#15151f;border-radius:10px;overflow:hidden;font-family:system-ui,sans-serif">
  <div id="stage" style="flex:1;min-height:0;background:radial-gradient(circle at 50% 40%, #3a3f4b 0%, #1b1d24 75%)"></div>
  <div id="controls" style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 12px;color:#e3f2fd;font-size:13px;background:#15151f">
    <span>Antes</span>
    <input id="morph" type="range" min="0" max="1" step="0.01" value="1" style="flex:1;min-width:120px">
    <span>Después</span>
    <button data-view="left">3/4 izq.</button>
    <button data-view="front">Frente</button>
    <button data-view="right">3/4 der.</button>
    <label><input id="tex" type="checkbox" checked> Textura</label>
    <label><input id="bg" type="checkbox" checked> Pelo y cuello</label>
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
geometry.setAttribute("color", new THREE.Float32BufferAttribute(
  DATA.alpha.flatMap((a) => [1, 1, 1, a]), 4
));
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
const textured = new THREE.MeshStandardMaterial({ map: texture, roughness: 1, metalness: 0, side: THREE.DoubleSide, alphaTest: 0.5 });
const plain = new THREE.MeshStandardMaterial({ color: 0xd9b8a3, roughness: 0.6, side: THREE.DoubleSide });
// Pelo y cuello: el borde exterior se desvanece (opacidad por vértice)
const texturedBg = new THREE.MeshStandardMaterial({
  map: texture, roughness: 1, metalness: 0, side: THREE.DoubleSide, vertexColors: true, transparent: true,
  alphaTest: 0.03,
});
const plainBg = new THREE.MeshStandardMaterial({
  color: 0xd9b8a3, roughness: 0.6, side: THREE.DoubleSide, vertexColors: true, transparent: true,
});
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
// Rotación limitada: una foto frontal no tiene datos de los laterales de la cabeza
const MAX_YAW = THREE.MathUtils.degToRad(40);
const MAX_PITCH = THREE.MathUtils.degToRad(20);
controls.minAzimuthAngle = -MAX_YAW;
controls.maxAzimuthAngle = MAX_YAW;
controls.minPolarAngle = Math.PI / 2 - MAX_PITCH;
controls.maxPolarAngle = Math.PI / 2 + MAX_PITCH;
controls.enablePan = false;

const views = { left: -32, front: 0, right: 32 };   // grados de giro
function setView(name) {
  const d = fitDistance();
  const yaw = THREE.MathUtils.degToRad(views[name]);
  camera.position.set(target.x + Math.sin(yaw) * d, target.y + 0.04 * d, target.z + Math.cos(yaw) * d);
  controls.target.copy(target);
  controls.minDistance = 0.6 * d;
  controls.maxDistance = 1.4 * d;
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
