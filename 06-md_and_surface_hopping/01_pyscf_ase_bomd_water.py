"""Module 06 Example 01: Ab Initio Born-Oppenheimer Molecular Dynamics (BOMD) of Water."""
from __future__ import annotations
import numpy as np
from mace_md_pipeline import water_anharmonic_energy_and_forces


def run_water_nve_bomd(
    n_steps: int = 100,
    dt_fs: float = 0.25,
) -> dict[str, np.ndarray | float]:
    """Run microcanonical (NVE) Velocity Verlet BOMD of H2O and verify energy conservation."""
    ev_to_eh = 1.0 / 27.211386245988
    ang_to_bohr = 1.0 / 0.529177210903
    fs_to_au = 41.341373335182

    masses_amu = np.array([15.994915, 1.007825, 1.007825])
    masses_au = masses_amu * 1822.888486
    dt_au = dt_fs * fs_to_au

    r0_ang = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.785, 0.0, 0.605],
            [-0.785, 0.0, 0.605],
        ],
        dtype=np.float64,
    )
    pos_bohr = r0_ang * ang_to_bohr
    vel_au = np.zeros_like(pos_bohr)

    def get_pe_and_forces_au(r_b: np.ndarray) -> tuple[float, np.ndarray]:
        r_ang = r_b / ang_to_bohr
        e_ev, f_ev_ang = water_anharmonic_energy_and_forces(r_ang)
        return e_ev * ev_to_eh, f_ev_ang * (ev_to_eh / ang_to_bohr)

    pe, forces = get_pe_and_forces_au(pos_bohr)
    e_tot_hist = np.zeros(n_steps + 1)
    e_tot_hist[0] = pe

    for step in range(1, n_steps + 1):
        vel_half = vel_au + 0.5 * (forces / masses_au[:, None]) * dt_au
        pos_bohr = pos_bohr + vel_half * dt_au
        pe, forces = get_pe_and_forces_au(pos_bohr)
        vel_au = vel_half + 0.5 * (forces / masses_au[:, None]) * dt_au
        ke = 0.5 * float(np.sum(masses_au[:, None] * (vel_au**2)))
        e_tot_hist[step] = ke + pe

    max_drift = float(np.max(np.abs(e_tot_hist - e_tot_hist[0])))
    return {"e_total_history": e_tot_hist, "max_energy_drift_eh": max_drift}


if __name__ == "__main__":
    res = run_water_nve_bomd(n_steps=100, dt_fs=0.25)
    print(f"H2O 3D Velocity Verlet BOMD max NVE energy drift: {res['max_energy_drift_eh']:.2e} Eh")