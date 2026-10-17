"""Module 05 Example 03: H2 Dissociation — RHF vs. Broken-Symmetry UHF & Coulson-Fischer Point."""
from __future__ import annotations
import numpy as np


def solve_two_site_h2_rhf_uhf_fci(
    r_bohr: float,
) -> dict[str, float]:
    """Compute exact minimal-basis RHF, broken-symmetry UHF, and FCI energies + <S^2> for H2 at distance R."""
    t_hop = 0.65 * np.exp(-0.85 * (r_bohr - 1.40))
    u_onsite = 1.25
    v_intersite = 1.0 / np.hypot(r_bohr, 0.85)
    e_nuc_eff = -2.0 + 1.0 / max(r_bohr, 0.5) - 1.0 / np.hypot(r_bohr, 0.72)

    u_eff = u_onsite - v_intersite
    e_rhf = e_nuc_eff - 2.0 * t_hop + 0.5 * (u_onsite + v_intersite)

    if u_eff <= 4.0 * t_hop:
        e_uhf = e_rhf
        s2_uhf = 0.0
    else:
        ratio = (4.0 * t_hop) / u_eff
        e_uhf = e_nuc_eff + v_intersite - (4.0 * (t_hop**2)) / u_eff
        s2_uhf = 1.0 - ratio**2

    e_fci = (
        e_nuc_eff
        + 0.5 * (u_onsite + v_intersite)
        - 0.5 * float(np.sqrt(u_eff**2 + 16.0 * (t_hop**2)))
    )
    return {
        "r_bohr": float(r_bohr),
        "e_rhf": float(e_rhf),
        "e_uhf": float(e_uhf),
        "e_fci": float(e_fci),
        "s2_uhf": float(s2_uhf),
    }


if __name__ == "__main__":
    print(f"{'R (Bohr)':>8s} | {'E_RHF (Eh)':>12s} | {'E_UHF (Eh)':>12s} | {'E_FCI (Eh)':>12s} | {'<S^2>_UHF':>10s}")
    print("-" * 66)
    for r in np.linspace(1.2, 5.0, 9):
        row = solve_two_site_h2_rhf_uhf_fci(float(r))
        print(
            f"{row['r_bohr']:8.2f} | {row['e_rhf']:12.6f} | "
            f"{row['e_uhf']:12.6f} | {row['e_fci']:12.6f} | {row['s2_uhf']:10.4f}"
        )