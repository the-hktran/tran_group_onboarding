"""Module 07A Project (Student Starter with #TODOs): MACE extxyz Pipeline & Committee Uncertainty."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class AtomicConfiguration:
    symbols: list[str]
    positions: np.ndarray
    energy_ev: float
    forces_ev_ang: np.ndarray
    config_type: str = "MD_Trajectory"


def water_anharmonic_energy_and_forces(
    positions: np.ndarray,
    e0_o: float = -2041.80,
    e0_h: float = -13.60,
    k_r: float = 36.5,
    r0: float = 0.9572,
    k_theta: float = 4.35,
    theta0_deg: float = 104.52,
) -> tuple[float, np.ndarray]:
    """Evaluate a smooth anharmonic H2O potential (eV) and analytical forces (eV/Angstrom)."""
    # TODO 1: Compute O-H bond vectors v1 = r_H1 - r_O, v2 = r_H2 - r_O, bond lengths d1, d2,
    #         and bond angle theta = arccos(u1 . u2). Evaluate the anharmonic stretching +
    #         bending + stretch-stretch coupling energy E_total and analytical forces F_i = -dE/dR_i.
    raise NotImplementedError("Complete TODO 1 in water_anharmonic_energy_and_forces")


def write_mace_extxyz(configs: list[AtomicConfiguration], filepath: str | Path) -> None:
    """Write configurations in MACE-compatible Extended XYZ (extxyz) format."""
    # TODO 2: Format each configuration with:
    #         Line 1: number of atoms
    #         Line 2: Properties=species:S:1:pos:R:3:REF_forces:R:3 REF_energy=<E> config_type=<type> pbc="F F F"
    #         Lines 3..N+2: <symbol> <x> <y> <z> <fx> <fy> <fz>
    raise NotImplementedError("Complete TODO 2 in write_mace_extxyz")


def compute_committee_force_uncertainty(
    committee_forces: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute per-atom and maximum committee force standard deviation sigma_F (eV/Ang)."""
    # TODO 3: Given committee_forces of shape (M_models, N_atoms, 3), compute the ensemble mean
    #         force vector on each atom, the per-atom RMS force deviation sigma_{F,i}, and
    #         return (atom_sigma_f, max(atom_sigma_f)) for active-learning query-by-committee.
    raise NotImplementedError("Complete TODO 3 in compute_committee_force_uncertainty")