"""Module 06 Example 02: Visualizing Diabatic vs. Adiabatic Surfaces & NACV Spikes."""
from __future__ import annotations
import numpy as np
from fssh_tully_solver import tully_model_1_adiabatic


def scan_avoided_crossing(
    r_min: float = -4.0, r_max: float = 4.0, n_points: int = 81
) -> dict[str, np.ndarray]:
    """Scan Tully Model I across the avoided crossing at R = 0 and tabulate E_0, E_1, and d_01(R)."""
    r_grid = np.linspace(r_min, r_max, n_points)
    e0_arr = np.zeros(n_points)
    e1_arr = np.zeros(n_points)
    d01_arr = np.zeros(n_points)

    for idx, r in enumerate(r_grid):
        energies, _, d01, _ = tully_model_1_adiabatic(float(r))
        e0_arr[idx] = energies[0]
        e1_arr[idx] = energies[1]
        d01_arr[idx] = d01

    return {
        "r_grid": r_grid,
        "e0_adiabatic": e0_arr,
        "e1_adiabatic": e1_arr,
        "nacv_d01": d01_arr,
    }


if __name__ == "__main__":
    scan = scan_avoided_crossing(-3.0, 3.0, 13)
    print(f"{'R (Bohr)':>9s} | {'E_0 (Eh)':>12s} | {'E_1 (Eh)':>12s} | {'Gap (Eh)':>10s} | {'d_01 (Bohr^-1)':>14s}")
    print("-" * 68)
    for r, e0, e1, d01 in zip(
        scan["r_grid"], scan["e0_adiabatic"], scan["e1_adiabatic"], scan["nacv_d01"]
    ):
        print(f"{r:9.2f} | {e0:12.6f} | {e1:12.6f} | {e1 - e0:10.6f} | {d01:14.6f}")