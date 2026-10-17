"""Module 04B Project (Student Starter with #TODOs): Standalone Pulay DIIS Manager."""
from __future__ import annotations
import numpy as np


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
        # TODO 1: Compute the orthonormalized commutator residual err = X.T @ (F @ D @ S - S @ D @ F) @ X
        #         and its RMS norm rms_err = sqrt(mean(err^2)).
        # TODO 2: Append F.copy() and err.copy() to self.fock_history and self.error_history,
        #         evicting the oldest entry if len > self.max_space.
        # TODO 3: If iteration >= self.start_iter and len >= 2, solve the Pulay DIIS linear system
        #         and return (F_extrap, rms_err); otherwise return (F.copy(), rms_err).
        raise NotImplementedError("Complete TODO 1-3 in PulayDIIS.update")