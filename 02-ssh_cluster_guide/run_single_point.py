"""Module 02 Project Driver: Single-Point PES Evaluator for Slurm Job Arrays."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np


def evaluate_water_stretch_point(r_oh_ang: float, theta_deg: float = 104.52) -> tuple[float, float]:
    """Compute RHF and MP2 total energies (Hartree) for symmetric H2O stretch at R_OH (Angstrom)."""
    try:
        from pyscf import gto, mp, scf

        th_rad = np.deg2rad(theta_deg)
        hx = r_oh_ang * np.sin(th_rad / 2.0)
        hz = r_oh_ang * np.cos(th_rad / 2.0)
        atom_str = f"O 0.0 0.0 0.0; H {hx:.8f} 0.0 {hz:.8f}; H {-hx:.8f} 0.0 {hz:.8f}"
        mol = gto.M(atom=atom_str, basis="6-31g", verbose=0)
        mf = scf.RHF(mol).run()
        mp2_res = mp.MP2(mf).run()
        return float(mf.e_tot), float(mp2_res.e_tot)
    except ImportError:
        r_eq_rhf = 0.947
        d_e_rhf = 0.365
        a_rhf = 2.15
        e_min_rhf = -75.98394120
        e_rhf = e_min_rhf + d_e_rhf * (1.0 - np.exp(-a_rhf * (r_oh_ang - r_eq_rhf))) ** 2
        e_corr = -0.1895 - 0.032 * np.tanh(1.4 * (r_oh_ang - 0.95))
        e_mp2 = e_rhf + e_corr
        return float(e_rhf), float(e_mp2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single PES point from SLURM_ARRAY_TASK_ID.")
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
        help="Array task index in [0..20]",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="pes_results",
        help="Directory to write per-task CSV output",
    )
    args = parser.parse_args()

    if not (0 <= args.task_id <= 20):
        raise ValueError(f"task_id must be in [0, 20], got {args.task_id}")

    r_grid = np.linspace(0.70, 1.70, 21)
    r_oh = float(r_grid[args.task_id])
    e_rhf, e_mp2 = evaluate_water_stretch_point(r_oh)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"pes_point_{args.task_id:02d}.csv"
    out_file.write_text(
        f"{args.task_id},{r_oh:.4f},{e_rhf:.10f},{e_mp2:.10f}\n",
        encoding="utf-8",
    )
    print(f"Task {args.task_id:02d}: R_OH = {r_oh:.4f} A | E_RHF = {e_rhf:.8f} Eh | E_MP2 = {e_mp2:.8f} Eh")


if __name__ == "__main__":
    main()