"""
simulation/viewer_glb.py — Visor 3D de modelos GLB (fal.ai) antes/después.

Muestra la cabeza completa generada en fal.ai con giro libre de 360°,
iluminación de estudio y botones para alternar Antes / Después.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

import base64
import json
from typing import Optional


THREE_VERSION = "0.160.0"


def build_glb_viewer_html(
    before_glb: Optional[bytes],
    after_glb: Optional[bytes],
    height: int = 560,
) -> str:
    """HTML autocontenido del visor, para `st.iframe`."""
    models = {
        name: base64.b64encode(glb).decode("ascii")
        for name, glb in (("before", before_glb), ("after", after_glb))
        if glb
    }
    if not models:
        raise ValueError("Se necesita al menos un modelo GLB.")

    return (
        _TEMPLATE
        .replace("__HEIGHT__", str(height))
        .replace("__THREE__", THREE_VERSION)
        .replace("__MODELS__", json.dumps(models))
    )


_TEMPLATE = """
<meta charset="utf-8">
<div id="wrap" style="width:100%;height:__HEIGHT__px;display:flex;flex-direction:column;border-radius:10px;overflow:hidden;font-family:system-ui,sans-serif">
  <div id="stage" style="flex:1;min-height:0;position:relative;background:radial-gradient(circle at 50% 40%, #3a3f4b 0%, #1b1d24 75%)">
    <div id="msg" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#e3f2fd;font-size:14px">Cargando modelo 3D…</div>
  </div>
  <div id="controls" style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 12px;color:#e3f2fd;font-size:13px;background:#15151f">
    <button data-model="before">Antes</button>
    <button data-model="after">Después</button>
    <label><input id="spin" type="checkbox"> Girar automáticamente</label>
  </div>
</div>
<style>
  body { margin: 0 }
  #controls button { background:#37474f;color:#fff;border:0;border-radius:6px;padding:4px 12px;cursor:pointer }
  #controls button.active { background:#1565c0 }
  #controls button:disabled { opacity:.35;cursor:default }
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
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const MODELS = __MODELS__;
const stage = document.getElementById("stage");
const msg = document.getElementById("msg");

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(stage.clientWidth, stage.clientHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
const camera = new THREE.PerspectiveCamera(30, stage.clientWidth / stage.clientHeight, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Escala común: ambos modelos se normalizan al mismo tamaño y centro
function normalize(root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const scale = 1 / Math.max(size.x, size.y, size.z);
  root.position.sub(center.multiplyScalar(scale));
  root.scale.setScalar(scale);
}

function base64ToBuffer(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

const loader = new GLTFLoader();
const loaded = {};
await Promise.all(Object.entries(MODELS).map(([name, b64]) =>
  new Promise((resolve) => {
    loader.parse(base64ToBuffer(b64), "", (gltf) => {
      normalize(gltf.scene);
      gltf.scene.visible = false;
      scene.add(gltf.scene);
      loaded[name] = gltf.scene;
      resolve();
    }, (err) => { console.error(err); resolve(); });
  })
));
msg.remove();

const buttons = document.querySelectorAll("#controls button");
function show(name) {
  for (const [key, obj] of Object.entries(loaded)) obj.visible = key === name;
  buttons.forEach((b) => b.classList.toggle("active", b.dataset.model === name));
}
buttons.forEach((b) => {
  b.disabled = !loaded[b.dataset.model];
  b.addEventListener("click", () => show(b.dataset.model));
});

camera.position.set(0, 0.05, 2.2);
controls.target.set(0, 0, 0);
controls.update();
show(loaded.after ? "after" : "before");

document.getElementById("spin").addEventListener("change", (e) => {
  controls.autoRotate = e.target.checked;
});
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
