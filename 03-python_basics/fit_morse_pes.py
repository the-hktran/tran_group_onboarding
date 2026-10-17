"""Module 03B Project: Non-Linear Least Squares Morse PES Fitting & Spectroscopic Constants."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit


@dataclass
class MorseFitResult:
    d_e_hartree: float
    a_inv_bohr: float
    r_eq_bohr: float
    e_min_hartree: float
    harmonic_k_au: float
    omega_e_cm1: float
    rmse_hartree: float


def morse_potential(
    r_bohr: np.ndarray,
    d_e: float,
    a: float,
    r_eq: float,
    e_min: float,
) -> np.ndarray:
    """Evaluate shifted Morse potential E(R) = E_min + D_e * (1 - exp(-a*(R - R_eq)))^2."""
    return e_min + d_e * (1.0 - np.exp(-a * (r_bohr - r_eq))) ** 2


def fit_morse_curve(
    r_bohr: np.ndarray,
    energies_eh: np.ndarray,
    reduced_mass_amu: float = 0.50391,
) -> MorseFitResult:
    """Fit diatomic potential energy curve to a Morse potential and extract spectroscopic constants."""
    idx_min = int(np.argmin(energies_eh))
    e_min_guess = float(energies_eh[idx_min])
    r_eq_guess = float(r_bohr[idx_min])
    d_e_guess = max(0.05, float(energies_eh[-1] - e_min_guess))
    a_guess = 1.0

    popt, _ = curve_fit(
        morse_potential,
        r_bohr,
        energies_eh,
        p0=[d_e_guess, a_guess, r_eq_guess, e_min_guess],
        maxfev=10000,
    )
    d_e_fit, a_fit, r_eq_fit, e_min_fit = [float(x) for x in popt]

    k_au = 2.0 * d_e_fit * (a_fit**2)
    mu_au = reduced_mass_amu * 1822.888486
    omega_au = float(np.sqrt(k_au / mu_au))
    omega_cm1 = omega_au * 219474.63137

    pred = morse_potential(r_bohr, d_e_fit, a_fit, r_eq_fit, e_min_fit)
    rmse = float(np.sqrt(np.mean((pred - energies_eh) ** 2)))

    return MorseFitResult(
        d_e_hartree=d_e_fit,
        a_inv_bohr=a_fit,
        r_eq_bohr=r_eq_fit,
        e_min_hartree=e_min_fit,
        harmonic_k_au=k_au,
        omega_e_cm1=omega_cm1,
        rmse_hartree=rmse,
    )


if __name__ == "__main__":
    r_grid = np.linspace(0.9, 4.5, 25)
    e_synth = morse_potential(r_grid, d_e=0.1745, a=1.0282, r_eq=1.4011, e_min=-1.1745)
    fit = fit_morse_curve(r_grid, e_synth, reduced_mass_amu=0.50391)
    print(f"Fitted R_eq:    {fit.r_eq_bohr:.4f} Bohr (expected 1.4011)")
    print(f"Fitted D_e:     {fit.d_e_hartree:.4f} Eh   (expected 0.1745)")
    print(f"Harmonic nu_e:  {fit.omega_e_cm1:.1f} cm^-1 (H2 expt ~4401 cm^-1)")
    print(f"Fit RMSE:       {fit.rmse_hartree:.2e} Eh")