"""Module 04C Bonus (Student Starter with #TODOs): Canonical MP2 & Grimme SCS-MP2."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MP2Result:
    e_corr_same_spin: float
    e_corr_opposite_spin: float
    e_corr_mp2: float
    e_corr_scs_mp2: float
    t2_amplitudes: np.ndarray


def run_mp2_detailed(
    eri_ao: np.ndarray,
    C: np.ndarray,
    eps: np.ndarray,
    n_occ: int,
    c_os: float = 1.20,
    c_ss: float = 1.0 / 3.0,
) -> MP2Result:
    """Compute canonical MP2 and Grimme Spin-Component-Scaled (SCS-MP2) correlation energies."""
    # TODO 1: Transform eri_ao to MO integrals iajb = (ia|jb) using C_occ = C[:, :n_occ]
    #         and C_virt = C[:, n_occ:] via np.einsum(..., optimize=True).
    # TODO 2: Construct the 4D energy denominator D_{iajb} = eps_i - eps_a + eps_j - eps_b
    #         and first-order amplitudes t2 = iajb / D_{iajb}.
    # TODO 3: Separate opposite-spin correlation e_os = sum(t2 * iajb) and same-spin
    #         correlation e_ss = sum(t2 * (iajb - ibja)), then compute e_mp2 = e_os + e_ss
    #         and Grimme SCS-MP2 energy e_scs = c_os * e_os + c_ss * e_ss.
    raise NotImplementedError("Complete TODO 1-3 in run_mp2_detailed")