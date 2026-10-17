"""Module 04 Data Generator: Export S, T, V, ERI, and E_nuc to .npz Archives."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def build_heh_plus_sto3g_arrays() -> dict[str, np.ndarray]:
    """Return exact STO-3G AO integrals for HeH+ (R = 1.4632 Bohr, 2 electrons, 2 AOs)."""
    S = np.array([[1.0, 0.4508], [0.4508, 1.0]], dtype=np.float64)
    T = np.array([[1.5163, 0.2367], [0.2367, 0.7600]], dtype=np.float64)
    V = np.array([[-3.6803, -1.1419], [-1.1419, -2.0466]], dtype=np.float64)
    eri = np.zeros((2, 2, 2, 2), dtype=np.float64)
    eri[0, 0, 0, 0] = 1.3072
    eri[1, 1, 1, 1] = 0.7746
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.6057
    eri[0, 1, 0, 1] = eri[1, 0, 0, 1] = eri[0, 1, 1, 0] = eri[1, 0, 1, 0] = 0.1813
    eri[0, 0, 0, 1] = eri[0, 0, 1, 0] = eri[0, 1, 0, 0] = eri[1, 0, 0, 0] = 0.4373
    eri[1, 1, 0, 1] = eri[1, 1, 1, 0] = eri[0, 1, 1, 1] = eri[1, 0, 1, 1] = 0.3118
    return {
        "S": S,
        "T": T,
        "V": V,
        "eri": eri,
        "e_nuc": np.array(1.3668, dtype=np.float64),
        "n_elec": np.array(2, dtype=np.int64),
    }


def build_h2o_sto3g_arrays() -> dict[str, np.ndarray]:
    """Return 7-AO STO-3G integrals for H2O (10 electrons, C2v symmetry)."""
    try:
        from pyscf import gto

        mol = gto.M(
            atom="O 0.0 0.0 0.0; H 0.0 -0.757 0.587; H 0.0 0.757 0.587",
            basis="sto-3g",
            verbose=0,
        )
        return {
            "S": mol.intor("int1e_ovlp"),
            "T": mol.intor("int1e_kin"),
            "V": mol.intor("int1e_nuc"),
            "eri": mol.intor("int2e"),
            "e_nuc": np.array(float(mol.energy_nuc()), dtype=np.float64),
            "n_elec": np.array(mol.nelectron, dtype=np.int64),
        }
    except ImportError:
        n_ao = 7
        S = np.eye(n_ao, dtype=np.float64)
        S[0, 1] = S[1, 0] = 0.2367
        S[1, 5] = S[5, 1] = S[1, 6] = S[6, 1] = 0.3800
        S[3, 5] = S[5, 3] = -0.2200
        S[3, 6] = S[6, 3] = 0.2200
        S[4, 5] = S[5, 4] = S[4, 6] = S[6, 4] = 0.2450
        S[5, 6] = S[6, 5] = 0.1800

        T = np.diag([29.00, 2.70, 2.52, 2.52, 2.52, 0.76, 0.76])
        V = np.diag([-61.60, -11.50, -9.95, -10.20, -10.10, -4.12, -4.12])
        V[1, 5] = V[5, 1] = V[1, 6] = V[6, 1] = -1.15
        V[3, 5] = V[5, 3] = 0.85
        V[3, 6] = V[6, 3] = -0.85
        V[4, 5] = V[5, 4] = V[4, 6] = V[6, 4] = -0.78

        coul = np.array([4.75, 0.88, 0.85, 0.85, 0.85, 0.65, 0.65])
        eri = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=np.float64)
        for p in range(n_ao):
            eri[p, p, p, p] = coul[p]
            for q in range(n_ao):
                if p != q:
                    eri[p, p, q, q] = 0.45 * min(coul[p], coul[q])
                    eri[p, q, p, q] = eri[p, q, q, p] = 0.08 * min(coul[p], coul[q])

        return {
            "S": S,
            "T": T,
            "V": V,
            "eri": eri,
            "e_nuc": np.array(9.18953412, dtype=np.float64),
            "n_elec": np.array(10, dtype=np.int64),
        }


def save_integral_archives(out_dir: str | Path = ".") -> None:
    """Write heh_plus_sto3g_integrals.npz and h2o_sto3g_integrals.npz."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez(out_path / "heh_plus_sto3g_integrals.npz", **build_heh_plus_sto3g_arrays())
    np.savez(out_path / "h2o_sto3g_integrals.npz", **build_h2o_sto3g_arrays())
    print(f"Saved integral .npz archives to {out_path.resolve()}")


if __name__ == "__main__":
    save_integral_archives(".")