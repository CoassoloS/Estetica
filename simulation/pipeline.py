"""
simulation/pipeline.py — Orquestación: parámetros → malla → warp → re-medición.

Autor: FacialMetrics Pro
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from landmarks import AnthropometricPoint, LandmarkResult
from measurements import CalibrationResult, FacialAnalyzer, run_full_analysis
from simulation.mesh import FaceMesh3D
from simulation.procedures import compute_displacement
from simulation.warp import warp_image


@dataclass
class MeasurementDelta:
    name: str
    unit: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before


@dataclass
class SimulationOutcome:
    mesh_before: FaceMesh3D
    mesh_after: FaceMesh3D
    image: np.ndarray
    applied_fraction: float
    deltas: list[MeasurementDelta]


def run_simulation(
    image: np.ndarray,
    landmarks: LandmarkResult,
    calibration: CalibrationResult,
    values: dict[str, float],
    image_kind: str = "rest",
) -> SimulationOutcome:
    """Aplicar los procedimientos de `image_kind` sobre la foto y re-medir."""
    mesh_before = FaceMesh3D.from_landmarks(landmarks)
    displacement = compute_displacement(
        mesh_before, values, calibration.pixel_per_mm, image=image_kind
    )
    warp = warp_image(image, mesh_before, displacement)
    # Si el warp tuvo que limitarse, la malla refleja lo que realmente se ve
    mesh_after = mesh_before.displaced(displacement * warp.applied_fraction)

    return SimulationOutcome(
        mesh_before=mesh_before,
        mesh_after=mesh_after,
        image=warp.image,
        applied_fraction=warp.applied_fraction,
        deltas=measurement_deltas(mesh_before, mesh_after, landmarks, calibration),
    )


def measurement_deltas(
    mesh_before: FaceMesh3D,
    mesh_after: FaceMesh3D,
    reference: LandmarkResult,
    calibration: CalibrationResult,
) -> list[MeasurementDelta]:
    """Mediciones clínicas antes/después calculadas con el pipeline existente."""
    mm_px = calibration.mm_per_pixel
    AP = AnthropometricPoint

    def frontal(mesh: FaceMesh3D) -> dict[str, float]:
        lm = mesh.to_landmark_result(reference)
        report = run_full_analysis(lm, calibration, view_type="frontal")
        t = report.thirds
        return {
            "Altura facial total": t.total_height_mm,
            "Tercio superior": t.upper.height_mm,
            "Tercio medio": t.middle.height_mm,
            "Tercio inferior": t.lower.height_mm,
            "Labio superior (Sn–Sts)": t.lower_upper_lip_mm,
            "Mentón (Sti–Me)": t.lower_chin_mm,
            "Ancho nasal (Al–Al)": lm.distance_px(AP.ALAR_L, AP.ALAR_R) * mm_px,
            "Ancho bucal (Ch–Ch)": lm.distance_px(AP.CHEILION_L, AP.CHEILION_R) * mm_px,
            "Ancho bigonial aprox.": float(np.linalg.norm(mesh.xy[172] - mesh.xy[397])) * mm_px,
        }

    def profile(mesh: FaceMesh3D) -> dict[str, float]:
        lm = mesh.to_profile_landmark_result(reference)
        p = FacialAnalyzer().compute_profile(lm, mm_px)
        return {
            "Proyección nasal (perfil 3D)": p.nasal_projection_mm,
            "Proyección de mentón (perfil 3D)": p.chin_projection_mm,
            "Proyección labio inferior (perfil 3D)": p.lower_lip_projection_mm,
        }

    before = {**frontal(mesh_before), **profile(mesh_before)}
    after = {**frontal(mesh_after), **profile(mesh_after)}
    return [MeasurementDelta(name, "mm", before[name], after[name]) for name in before]
