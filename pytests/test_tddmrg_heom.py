"""Automated Pytest verification suite for Module 08 (TD-DMRG / MPS & HEOM Open Quantum Systems)."""
import numpy as np
from scipy.linalg import expm
from tddmrg_mps_solver import run_tddmrg_transport
from heom_spin_boson_solver import run_heom_spin_boson


def test_tddmrg_matches_exact_single_particle_chain() -> None:
    """Verify MPS TD-DMRG/TEBD transport against exact tight-binding unitary propagation."""
    n_sites = 5
    excited = 2
    j_hop = 1.0
    dt = 0.02
    n_steps = 40

    res_mps = run_tddmrg_transport(
        n_sites=n_sites,
        excited_site=excited,
        j_hop=j_hop,
        v_int=0.0,
        dt=dt,
        n_steps=n_steps,
        max_bond_dim=8,
    )

    # Exact 1-exciton subspace Hamiltonian H_{ij} = -j_hop * delta_{|i-j|, 1}
    H_1p = np.zeros((n_sites, n_sites), dtype=np.complex128)
    for k in range(n_sites - 1):
        H_1p[k, k + 1] = -j_hop
        H_1p[k + 1, k] = -j_hop
    psi_exact = np.zeros(n_sites, dtype=np.complex128)
    psi_exact[excited] = 1.0
    psi_exact = expm(-1j * H_1p * (n_steps * dt)) @ psi_exact
    pops_exact = np.abs(psi_exact) ** 2

    assert np.max(np.abs(res_mps.norm_history - 1.0)) < 1e-10
    np.testing.assert_allclose(res_mps.site_populations[-1], pops_exact, atol=1e-4)
    assert res_mps.entanglement_entropy[-1] > 0.1


def test_heom_trace_purity_and_tier_convergence() -> None:
    """Verify HEOM trace conservation, purity bounds, and convergence with hierarchy depth N_tier."""
    res_t1 = run_heom_spin_boson(n_tiers=1, dt=0.02, n_steps=150)
    res_t6 = run_heom_spin_boson(n_tiers=6, dt=0.02, n_steps=150)
    res_t8 = run_heom_spin_boson(n_tiers=8, dt=0.02, n_steps=150)

    # 1. Exact trace conservation Tr[rho_0(t)] == 1
    assert np.max(np.abs(res_t6.trace_history - 1.0)) < 1e-12

    # 2. Purity 0.5 <= Tr[rho_0^2] <= 1.0 for a 2-level system
    assert np.all(res_t6.purity_history <= 1.0 + 1e-10)
    assert np.all(res_t6.purity_history >= 0.5 - 1e-10)

    # 3. Non-Markovian memory (tiers > 1) noticeably alters coherence compared to tier 1
    assert np.max(np.abs(res_t6.sigma_x - res_t1.sigma_x)) > 1e-3

    # 4. Hierarchy convergence: n_tiers=6 and n_tiers=8 agree to < 1e-6
    np.testing.assert_allclose(res_t6.sigma_x, res_t8.sigma_x, atol=1e-6)
    np.testing.assert_allclose(res_t6.sigma_z, res_t8.sigma_z, atol=1e-6)