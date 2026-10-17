"""Automated Pytest suite for Module 03 (Hückel) and Module 04 (RHF + DIIS)."""
import numpy as np
from huckel_solver import build_huckel_hamiltonian, solve_pi_system
from rhf_solver import run_rhf, compute_mp2_correlation, symmetric_orthogonalization


def test_benzene_huckel() -> None:
    H = build_huckel_hamiltonian(6, alpha=0.0, beta=-1.0, cyclic=True)
    res = solve_pi_system(H, n_electrons=6)
    assert np.isclose(res["total_pi_energy"], -8.0, atol=1e-10)
    assert np.isclose(res["homo_lumo_gap"], 2.0, atol=1e-10)


def test_two_electron_he_h_plus_model() -> None:
    """Verify RHF solver on a 2-electron, 2-basis-function model system (HeH+ STO-3G)."""
    S = np.array([[1.0, 0.4508], [0.4508, 1.0]])
    T = np.array([[1.5163, 0.2367], [0.2367, 0.7600]])
    V = np.array([[-3.6803, -1.1419], [-1.1419, -2.0466]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 1.3072
    eri[1, 1, 1, 1] = 0.7746
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.6057
    eri[0, 1, 0, 1] = eri[1, 0, 0, 1] = eri[0, 1, 1, 0] = eri[1, 0, 1, 0] = 0.1813
    eri[0, 0, 0, 1] = eri[0, 0, 1, 0] = eri[0, 1, 0, 0] = eri[1, 0, 0, 0] = 0.4373
    eri[1, 1, 0, 1] = eri[1, 1, 1, 0] = eri[0, 1, 1, 1] = eri[1, 0, 1, 1] = 0.3118
    e_nuc = 1.3668

    X = symmetric_orthogonalization(S)
    np.testing.assert_allclose(X.T @ S @ X, np.eye(2), atol=1e-12)

    res_diis = run_rhf(S, T, V, eri, e_nuc, n_elec=2, use_diis=True)
    res_nodiis = run_rhf(S, T, V, eri, e_nuc, n_elec=2, use_diis=False)

    assert res_diis.converged and res_nodiis.converged
    assert np.isclose(res_diis.e_total, res_nodiis.e_total, atol=1e-9)
    e_mp2 = compute_mp2_correlation(eri, res_diis.mo_coeff, res_diis.orbital_energies, n_occ=1)
    assert e_mp2 < 0.0