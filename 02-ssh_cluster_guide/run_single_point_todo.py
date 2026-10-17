"""Module 02 Project (Student Starter with #TODOs): Single-Point PES Evaluator."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np


def build_water_xyz_from_bond_length(r_oh_ang: float, theta_deg: float = 104.52) -> str:
    """Return a PySCF-compatible atom string for symmetric H2O with bond length r_oh_ang."""
    # TODO 1: Convert theta_deg from degrees to radians using np.deg2rad(...)
    #         and compute hx = r_oh_ang * sin(theta/2) and hz = r_oh_ang * cos(theta/2).
    #         Return "O 0.0 0.0 0.0; H <hx> 0.0 <hz>; H -<hx> 0.0 <hz>".
    raise NotImplementedError("Complete TODO 1 in build_water_xyz_from_bond_length")


def evaluate_water_stretch_point(r_oh_ang: float, theta_deg: float = 104.52) -> tuple[float, float]:
    """Compute RHF and MP2 total energies (Hartree) for symmetric H2O stretch at R_OH (Angstrom)."""
    # TODO 2: Build the H2O geometry string using build_water_xyz_from_bond_length(r_oh_ang, theta_deg).
    #         If PySCF is installed, run RHF and MP2 in the 6-31g basis and return (e_rhf, e_mp2).
    #         Otherwise, evaluate the analytical Morse + correlation surrogate model.
    raise NotImplementedError("Complete TODO 2 in evaluate_water_stretch_point")


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

    # TODO 3: Validate that args.task_id is in [0, 20], construct the 21-point grid
    #         r_grid = np.linspace(0.70, 1.70, 21), evaluate (e_rhf, e_mp2) at r_grid[args.task_id],
    #         and write "<task_id>,<r_oh>,<e_rhf>,<e_mp2>\n" to pes_point_<task_id:02d>.csv.
    raise NotImplementedError("Complete TODO 3 in main")


if __name__ == "__main__":
    main()