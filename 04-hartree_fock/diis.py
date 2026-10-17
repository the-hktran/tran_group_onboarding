"""Module 04B: Standalone Pulay DIIS (Direct Inversion in the Iterative Subspace) Accelerator."""
from __future__ import annotations
import numpy as np
from rhf_solver import compute_orbital_gradient, diis_extrapolate


class PulayDIIS:
    """Object-oriented Pulay DIIS extrapolation manager for SCF solvers (RHF, UHF, Kohn-Sham)."""

    def __init__(self, max_space: int = 6, start_iter: int = 2) -> None:
        self.max_space = max_space
        self.start_iter = start_iter
        self.fock_history: list[np.ndarray] = []
        self.error_history: list[np.ndarray] = []

    def update(
        self,
        F: np.ndarray,
        D: np.ndarray,
        S: np.ndarray,
        X: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, float]:
        """Record Fock matrix and orbital gradient residual e = X.T(FDS - SDF)X, returning (F_extrap, rms_error)."""
        err = compute_orbital_gradient(F, D, S, X)
        rms_err = float(np.sqrt(np.mean(err**2)))

        self.fock_history.append(F.copy())
        self.error_history.append(err.copy())
        if len(self.fock_history) > self.max_space:
            self.fock_history.pop(0)
            self.error_history.pop(0)

        if iteration >= self.start_iter and len(self.fock_history) >= 2:
            F_extrap = diis_extrapolate(self.fock_history, self.error_history)
            return F_extrap, rms_err
        return F.copy(), rms_err