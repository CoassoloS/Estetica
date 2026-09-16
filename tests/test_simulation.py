"""
test_simulation.py — Tests para la simulación (procedimientos, warp, pipeline).

Verifica:
    - Desplazamientos en mm → px, vértices fijados y restricción por línea bucal
    - Warp piecewise-affine: identidad, movimiento de píxeles y límite de pliegues
    - Re-medición antes/después
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measurements import CalibrationResult
from simulation import FaceMesh3D, PRESETS, default_values, describe_for_ai, run_simulation
from simulation.mesh import tessellation_triangles
from simulation.procedures import (
    LIP_INNER_LOWER,
    LIP_INNER_UPPER,
    PROCEDURES,
    compute_displacement,
    param_id,
)
from simulation.warp import warp_image
from tests.sim_fixtures import make_synthetic_landmarks, make_texture_image

PX_PER_MM = 3.0


@pytest.fixture
def landmarks():
    return make_synthetic_landmarks()


@pytest.fixture
def mesh(landmarks):
    return FaceMesh3D.from_landmarks(landmarks)


@pytest.fixture
def calibration():
    return CalibrationResult(
        mm_per_pixel=1 / PX_PER_MM, method="manual",
        reference_distance_mm=10, reference_distance_px=30,
    )


def values_with(**kwargs):
    values = default_values()
    for key, value in kwargs.items():
        values[key.replace("__", ".")] = value
    return values


# ──────────────────────────────────────────────────────────────────────────────
# Procedimientos
# ──────────────────────────────────────────────────────────────────────────────

class TestProcedures:

    def test_defaults_produce_no_displacement(self, mesh):
        disp = compute_displacement(mesh, default_values(), PX_PER_MM)
        assert np.allclose(disp, 0)

    def test_requires_calibration(self, mesh):
        with pytest.raises(ValueError):
            compute_displacement(mesh, values_with(menton__vertical=2), 0.0)

    def test_anchor_moves_exact_mm(self, mesh):
        disp = compute_displacement(mesh, values_with(menton__vertical=2), PX_PER_MM, "rest")
        assert disp[152, 1] == pytest.approx(2 * PX_PER_MM)
        assert disp[152, 0] == pytest.approx(0, abs=1e-9)

    def test_forward_is_negative_z(self, mesh):
        disp = compute_displacement(mesh, values_with(menton__avance=3), PX_PER_MM)
        assert disp[152, 2] == pytest.approx(-3 * PX_PER_MM)
        assert np.allclose(disp[:, :2], 0)

    def test_far_vertices_do_not_move(self, mesh):
        disp = compute_displacement(mesh, values_with(menton__vertical=2), PX_PER_MM)
        glabella = 9
        assert np.allclose(disp[glabella], 0)

    def test_upper_lip_keeps_mouth_line_and_lower_lip(self, mesh):
        disp = compute_displacement(mesh, values_with(labios__superior=2), PX_PER_MM)
        # En el plano de la foto la línea de cierre no se mueve (sí proyecta en z)
        assert np.allclose(disp[list(LIP_INNER_UPPER), :2], 0)
        assert np.allclose(disp[list(LIP_INNER_LOWER)], 0)
        assert disp[0, 1] < 0      # Borde del bermellón sube
        below = mesh.vertices[:, 1] > 290
        assert np.allclose(disp[below], 0)

    def test_smile_procedure_ignored_on_rest_photo(self, mesh):
        values = values_with(sonrisa__gingival=2)
        assert np.allclose(compute_displacement(mesh, values, PX_PER_MM, "rest"), 0)
        assert not np.allclose(compute_displacement(mesh, values, PX_PER_MM, "smile"), 0)

    def test_tip_rotation_moves_tip_up(self, mesh):
        disp = compute_displacement(mesh, values_with(nariz__rotacion=10), PX_PER_MM)
        assert disp[4, 1] < 0
        assert np.allclose(disp[2], 0)          # Subnasale fijado

    def test_alar_width_is_symmetric(self, mesh):
        disp = compute_displacement(mesh, values_with(nariz__alar=-2), PX_PER_MM)
        assert disp[129, 0] > 0 and disp[358, 0] < 0
        total = (mesh.vertices[358, 0] + disp[358, 0]) - (mesh.vertices[129, 0] + disp[129, 0])
        assert total == pytest.approx(30 - 2 * PX_PER_MM, abs=0.5)

    def test_presets_reference_valid_parameters(self):
        valid = set(default_values())
        for preset in PRESETS:
            assert set(preset.values) <= valid, preset.name

    def test_parameters_default_within_range(self):
        for proc in PROCEDURES:
            for par in proc.parameters:
                assert par.min_value <= 0 <= par.max_value, param_id(proc, par)

    def test_describe_for_ai(self):
        lines = describe_for_ai(values_with(nariz__alar=-1.5), ["blanqueamiento"])
        assert any("1.5 mm" in line and "narrow" in line for line in lines)
        assert any("whiter" in line for line in lines)


# ──────────────────────────────────────────────────────────────────────────────
# Malla
# ──────────────────────────────────────────────────────────────────────────────

class TestMesh:

    def test_tessellation_triangles(self):
        tris = tessellation_triangles()
        assert tris.shape[1] == 3
        assert len(tris) > 800
        assert tris.max() < 468

    def test_landmark_result_roundtrip(self, mesh, landmarks):
        result = mesh.to_landmark_result(landmarks)
        assert result.points == landmarks.points


# ──────────────────────────────────────────────────────────────────────────────
# Warp
# ──────────────────────────────────────────────────────────────────────────────

class TestWarp:

    def test_zero_displacement_is_identity(self, mesh):
        image = make_texture_image()
        result = warp_image(image, mesh, np.zeros_like(mesh.vertices))
        assert np.array_equal(result.image, image)
        assert result.applied_fraction == 1.0

    def test_pixels_follow_vertex(self, mesh):
        image = make_texture_image()
        disp = compute_displacement(mesh, values_with(menton__vertical=2), PX_PER_MM)
        result = warp_image(image, mesh, disp)
        assert result.applied_fraction == 1.0

        src = mesh.vertices[152, :2]
        dst = src + disp[152, :2]
        original_patch = image[int(src[1]) - 2:int(src[1]) + 3, int(src[0]) - 2:int(src[0]) + 3]
        moved_patch = result.image[int(dst[1]) - 2:int(dst[1]) + 3, int(dst[0]) - 2:int(dst[0]) + 3]
        assert np.abs(original_patch.astype(int) - moved_patch.astype(int)).mean() < 6

    def test_far_pixels_unchanged(self, mesh):
        image = make_texture_image()
        disp = compute_displacement(mesh, values_with(menton__vertical=2), PX_PER_MM)
        result = warp_image(image, mesh, disp)
        assert np.array_equal(result.image[:120], image[:120])

    def test_folding_is_limited(self, mesh):
        image = make_texture_image()
        disp = np.zeros_like(mesh.vertices)
        disp[13, 1] = 60        # Stomion superior atraviesa el labio inferior
        result = warp_image(image, mesh, disp)
        assert 0 < result.applied_fraction < 1


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestPipeline:

    def test_remeasure_reflects_change(self, landmarks, calibration):
        image = make_texture_image()
        outcome = run_simulation(image, landmarks, calibration, values_with(menton__vertical=3))
        deltas = {d.name: d for d in outcome.deltas}
        assert deltas["Tercio inferior"].delta == pytest.approx(3, abs=0.4)
        assert deltas["Tercio superior"].delta == pytest.approx(0, abs=0.01)

    def test_profile_projection_delta(self, landmarks, calibration):
        image = make_texture_image()
        outcome = run_simulation(image, landmarks, calibration, values_with(menton__avance=4))
        deltas = {d.name: d for d in outcome.deltas}
        assert deltas["Proyección de mentón (perfil 3D)"].delta == pytest.approx(4, abs=0.4)


# ──────────────────────────────────────────────────────────────────────────────
# Visor 3D
# ──────────────────────────────────────────────────────────────────────────────

class TestViewer:

    def test_html_contains_mesh_data(self, mesh):
        from simulation import build_viewer_html

        html = build_viewer_html(mesh, mesh, make_texture_image())
        assert '<meta charset="utf-8">' in html
        assert "__DATA__" not in html and "__THREE__" not in html
        assert "data:image/jpeg;base64," in html
