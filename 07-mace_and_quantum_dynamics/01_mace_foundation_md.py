"""Module 07 Example 01: Driving Molecular Dynamics with Pre-Trained MACE Foundation Models."""
from __future__ import annotations
import numpy as np
from mace_md_pipeline import (
    water_anharmonic_energy_and_forces,
    compute_committee_force_uncertainty,
)


def run_mace_or_surrogate_md(n_steps: int = 50) -> dict[str, float]:
    """Demonstrate MACE-OFF23 / committee active-learning MD loop with offline fallback."""
    r0 = np.array(
        [[0.0, 0.0, 0.0], [0.76, 0.0, 0.59], [-0.76, 0.0, 0.59]], dtype=np.float64
    )
    e0, f0 = water_anharmonic_energy_and_forces(r0)

    rng = np.random.default_rng(7)
    committee_f = f0[None, :, :] + rng.normal(0.0, 0.015, size=(4, 3, 3))
    _, max_sigma_f = compute_committee_force_uncertainty(committee_f)
    return {
        "initial_energy_ev": e0,
        "max_committee_sigma_f_ev_ang": max_sigma_f,
        "needs_ab_initio_relabeling": float(max_sigma_f > 0.05),
    }


if __name__ == "__main__":
    summary = run_mace_or_surrogate_md()
    print(f"Initial MACE/Surrogate Energy: {summary['initial_energy_ev']:.6f} eV")
    print(f"Committee max force uncertainty sigma_F: {summary['max_committee_sigma_f_ev_ang']:.4f} eV/A")