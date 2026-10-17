"""Module 06 Project (Student Starter with #TODOs): Velocity Verlet MD & Tully FSSH."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MDTrajectory:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray


@dataclass
class FSSHTrajectoryResult:
    final_position: float
    final_momentum: float
    active_state: int
    transmitted: bool
    n_hops: int
    n_frustrated_hops: int
    max_energy_drift: float
    norm_drift: float


def morse_potential_and_force(
    r: float, d_e: float = 0.1745, a: float = 1.0282, r_eq: float = 1.4011
) -> tuple[float, float]:
    """Compute Morse potential V(r) and force F(r) = -dV/dr in atomic units (H2 default)."""
    # TODO 1: Evaluate V(r) = d_e * (1 - exp(-a*(r - r_eq)))^2 and analytical force F(r) = -dV/dr.
    raise NotImplementedError("Complete TODO 1 in morse_potential_and_force")


def run_velocity_verlet_morse(
    r0: float = 1.80,
    v0: float = 0.0,
    mass: float = 918.57,
    dt: float = 2.0,
    n_steps: int = 500,
) -> MDTrajectory:
    """Propagate 1D classical nuclear dynamics on a Morse potential via Velocity Verlet."""
    # TODO 2: Implement the symplectic Velocity Verlet integrator:
    #         1. v_half = v + 0.5 * (F / mass) * dt
    #         2. r_new  = r + v_half * dt
    #         3. V_new, F_new = morse_potential_and_force(r_new)
    #         4. v_new  = v_half + 0.5 * (F_new / mass) * dt
    raise NotImplementedError("Complete TODO 2 in run_velocity_verlet_morse")


def tully_model_1_adiabatic(
    r: float,
    A: float = 0.01,
    B: float = 1.6,
    C: float = 0.005,
    D: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute adiabatic energies E_0, E_1, forces F_0, F_1, NAC d_01(R), and eigenvectors U(R)."""
    # TODO 3: Build Tully Model I diabatic matrix V_diab(r) and its derivative dH_diab(r),
    #         diagonalize V_diab via mixing angle theta = 0.5 * arctan2(V12, -delta),
    #         compute Hellmann-Feynman adiabatic forces F_k = -diag(U^T dH_diab U),
    #         and compute nonadiabatic coupling d_01(r) = (U^T dH_diab U)_{01} / (E_1 - E_0).
    raise NotImplementedError("Complete TODO 3 in tully_model_1_adiabatic")


def rk4_electronic_step(
    c: np.ndarray,
    energies: np.ndarray,
    v_dot_d01: float,
    dt: float,
    n_substeps: int = 10,
) -> np.ndarray:
    """Integrate i dc/dt = H_eff * c over dt using RK4 sub-stepping."""
    # TODO 4: Integrate dc_0/dt = -i E_0 c_0 - (v * d_01) c_1 and
    #                   dc_1/dt = -i E_1 c_1 + (v * d_01) c_0
    #         over `n_substeps` fourth-order Runge-Kutta (RK4) sub-steps of size dt / n_substeps.
    raise NotImplementedError("Complete TODO 4 in rk4_electronic_step")


def run_single_fssh_trajectory(
    k0: float = 20.0,
    r0: float = -8.0,
    mass: float = 2000.0,
    dt: float = 1.0,
    max_steps: int = 8000,
    seed: int = 42,
    apply_decoherence: bool = True,
) -> FSSHTrajectoryResult:
    """Propagate a single Tully FSSH trajectory on Tully Model I starting on state 0."""
    # TODO 5: Implement Tully Fewest-Switches Surface Hopping:
    #         1. Advance (r, v) on `active_state` using Velocity Verlet.
    #         2. Advance electronic amplitudes `c` using rk4_electronic_step.
    #         3. Evaluate switching probability g_{k->j} = max(0, dt * b_{jk} / |c_k|^2).
    #         4. If rng.random() < g_{k->j}, check if kinetic energy >= dE_pot:
    #            - If sufficient, rescale velocity v_new to conserve total energy and hop!
    #            - Otherwise, record a frustrated hop.
    #         5. Apply Granucci-Persico decoherence damping if apply_decoherence is True.
    raise NotImplementedError("Complete TODO 5 in run_single_fssh_trajectory")