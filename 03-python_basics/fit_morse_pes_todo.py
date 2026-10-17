"""Module 03B Project (Student Starter with #TODOs): Non-Linear Least Squares Morse PES Fitting."""
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
    # TODO 1: Implement the vectorized Morse potential formula in atomic units.
    raise NotImplementedError("Complete TODO 1 in morse_potential")


def fit_morse_curve(
    r_bohr: np.ndarray,
    energies_eh: np.ndarray,
    reduced_mass_amu: float = 0.50391,
) -> MorseFitResult:
    """Fit diatomic potential energy curve to a Morse potential and extract spectroscopic constants."""
    # TODO 2: Construct physical initial guesses [d_e_guess, a_guess, r_eq_guess, e_min_guess]
    #         from the minimum and asymptotic values of (r_bohr, energies_eh), then call
    #         scipy.optimize.curve_fit(morse_potential, r_bohr, energies_eh, p0=...).
    # TODO 3: Compute the harmonic force constant k_au = 2 * D_e * a^2, convert reduced_mass_amu
    #         to atomic units (1 amu = 1822.888486 m_e), evaluate omega_au = sqrt(k_au / mu_au),
    #         and convert to wavenumbers omega_cm1 = omega_au * 219474.63137.
    raise NotImplementedError("Complete TODO 2 & TODO 3 in fit_morse_curve")