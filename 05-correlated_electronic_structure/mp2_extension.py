"""Module 04C Bonus: Spin-Component-Scaled (SCS-MP2) and Canonical MP2 Correlation Energy."""
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
    """Compute canonical MP2 and Grimme's Spin-Component-Scaled (SCS-MP2) correlation energies."""
    C_occ = C[:, :n_occ]
    C_virt = C[:, n_occ:]
    iajb = np.einsum(
        "uvls,ui,va,lj,sb->iajb", eri_ao, C_occ, C_virt, C_occ, C_virt, optimize=True
    )
    ibja = np.swapaxes(iajb, 1, 3)

    e_occ = eps[:n_occ]
    e_virt = eps[n_occ:]
    denom = (
        e_occ[:, None, None, None]
        - e_virt[None, :, None, None]
        + e_occ[None, None, :, None]
        - e_virt[None, None, None, :]
    )
    t2 = iajb / denom

    e_os = float(np.sum(t2 * iajb))
    e_ss = float(np.sum(t2 * (iajb - ibja)))
    e_mp2 = e_os + e_ss
    e_scs = c_os * e_os + c_ss * e_ss

    return MP2Result(
        e_corr_same_spin=e_ss,
        e_corr_opposite_spin=e_os,
        e_corr_mp2=e_mp2,
        e_corr_scs_mp2=e_scs,
        t2_amplitudes=t2,
    )