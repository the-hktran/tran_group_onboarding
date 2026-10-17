"""Module 05 Example 02: Basis Set Convergence & Two-Point CBS Extrapolation (Helgaker Formula)."""
from __future__ import annotations


H2O_BASIS_BENCHMARK: list[dict[str, str | int | float]] = [
    {"basis": "STO-3G", "cardinal_X": 1, "n_ao": 7, "e_rhf": -74.96302314, "e_mp2_corr": -0.03551840},
    {"basis": "6-31G*", "cardinal_X": 2, "n_ao": 19, "e_rhf": -76.01074650, "e_mp2_corr": -0.18842110},
    {"basis": "cc-pVDZ", "cardinal_X": 2, "n_ao": 24, "e_rhf": -76.02676542, "e_mp2_corr": -0.20418930},
    {"basis": "cc-pVTZ", "cardinal_X": 3, "n_ao": 58, "e_rhf": -76.05712890, "e_mp2_corr": -0.26841205},
    {"basis": "cc-pVQZ", "cardinal_X": 4, "n_ao": 115, "e_rhf": -76.06478912, "e_mp2_corr": -0.28994180},
    {"basis": "aug-cc-pVTZ", "cardinal_X": 3, "n_ao": 92, "e_rhf": -76.06058140, "e_mp2_corr": -0.27981044},
]


def helgaker_cbs_extrapolation(
    x_small: int,
    e_corr_small: float,
    x_large: int,
    e_corr_large: float,
) -> float:
    """Extrapolate correlation energy to Complete Basis Set (CBS) limit via E(X) = E_CBS + A * X^(-3)."""
    x3_s = float(x_small**3)
    x3_l = float(x_large**3)
    return float((x3_l * e_corr_large - x3_s * e_corr_small) / (x3_l - x3_s))


if __name__ == "__main__":
    print(f"{'Basis Set':12s} | {'N_AO':>4s} | {'E_RHF (Eh)':>14s} | {'E_corr MP2 (Eh)':>15s} | {'E_tot MP2 (Eh)':>14s}")
    print("-" * 70)
    for row in H2O_BASIS_BENCHMARK:
        e_tot = float(row["e_rhf"]) + float(row["e_mp2_corr"])
        print(
            f"{str(row['basis']):12s} | {int(row['n_ao']):4d} | "
            f"{float(row['e_rhf']):14.8f} | {float(row['e_mp2_corr']):15.8f} | {e_tot:14.8f}"
        )

    e_cbs_corr = helgaker_cbs_extrapolation(
        3, float(H2O_BASIS_BENCHMARK[3]["e_mp2_corr"]),
        4, float(H2O_BASIS_BENCHMARK[4]["e_mp2_corr"]),
    )
    print(f"\nHelgaker CBS(T->Q) extrapolated MP2 E_corr: {e_cbs_corr:.8f} Eh")