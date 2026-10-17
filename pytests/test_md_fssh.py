"""Automated Pytest verification suite for Module 06 (Velocity Verlet MD & Tully FSSH)."""
import numpy as np
from fssh_tully_solver import (
    run_velocity_verlet_morse,
    tully_model_1_adiabatic,
    run_single_fssh_trajectory,
    run_fssh_ensemble,
)


def test_velocity_verlet_energy_conservation() -> None:
    """Verify symplectic NVE energy conservation of Velocity Verlet on a Morse oscillator."""
    traj = run_velocity_verlet_morse(r0=1.75, v0=0.0, dt=1.0, n_steps=500)
    energy_drift = np.max(np.abs(traj.total_energy - traj.total_energy[0]))
    assert energy_drift < 1e-6, f"Energy drift too large: {energy_drift}"


def test_analytical_vs_numerical_nacv() -> None:
    """Verify analytical nonadiabatic coupling d_01(R) against finite-difference <Phi_0(R)|Phi_1(R+dr)>/dr."""
    for r_val in (-1.2, -0.3, 0.0, 0.4, 1.5):
        dr = 1e-6
        _, _, _, U_minus = tully_model_1_adiabatic(r_val - dr)
        _, _, _, U_plus = tully_model_1_adiabatic(r_val + dr)
        _, _, d01_center, U_center = tully_model_1_adiabatic(r_val)

        # Ensure smooth eigenvector sign gauge
        for col in range(2):
            if np.dot(U_center[:, col], U_minus[:, col]) < 0:
                U_minus[:, col] *= -1.0
            if np.dot(U_center[:, col], U_plus[:, col]) < 0:
                U_plus[:, col] *= -1.0

        d_phi1_dr = (U_plus[:, 1] - U_minus[:, 1]) / (2.0 * dr)
        d01_num = float(np.dot(U_center[:, 0], d_phi1_dr))
        assert np.isclose(d01_center, d01_num, atol=1e-7)


def test_fssh_conservation_and_branching() -> None:
    """Verify norm conservation, energy conservation across surface hops, and high-k0 nonadiabatic transitions."""
    res = run_single_fssh_trajectory(k0=22.0, seed=7)
    assert res.max_energy_drift < 1e-5
    assert res.norm_drift < 1e-6

    low_k = run_fssh_ensemble(k0=9.0, n_traj=30)
    high_k = run_fssh_ensemble(k0=25.0, n_traj=30)
    # At higher momentum k0=25 a.u., nonadiabatic transmission on state 1 exceeds low-k0 transmission
    assert high_k["transmission_state_1"] > low_k["transmission_state_1"]