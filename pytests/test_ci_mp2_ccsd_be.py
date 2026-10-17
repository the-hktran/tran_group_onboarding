"""Automated Pytest verification suite for Module 05 (CI, MP2, CCSD & Bootstrap Embedding Matching Loops)."""
import numpy as np
from rhf_solver import run_rhf
from correlated_methods_ci_mp2_ccsd import (
    mo_eri_transform,
    run_mp2_from_mo,
    run_ccsd_spin_orbital,
    run_2e_singlet_fci,
    compare_size_extensivity_dimers,
)
from schmidt_bath_construction import (
    construct_schmidt_bath_from_c_occ,
    construct_schmidt_bath_from_1rdm,
    run_active_space_fci,
    run_bootstrap_embedding_cumulant,
    run_bootstrap_embedding_matching_loop,
)


def _heh_plus_sto3g_model() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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
    return S, T, V, eri, e_nuc


def test_ci_mp2_ccsd_equivalence_for_two_electrons() -> None:
    """Verify MP2, CCSD, and FCI on a 2-electron system (where CCSD == CISD == FCI exactly)."""
    S, T, V, eri, e_nuc = _heh_plus_sto3g_model()
    rhf = run_rhf(S, T, V, eri, e_nuc, n_elec=2)
    h_mo = rhf.mo_coeff.T @ (T + V) @ rhf.mo_coeff
    eri_mo = mo_eri_transform(eri, rhf.mo_coeff)

    e_mp2_corr, _, _ = run_mp2_from_mo(eri_mo, rhf.orbital_energies, n_occ=1)
    e_ccsd_corr, _, _ = run_ccsd_spin_orbital(h_mo, eri_mo, rhf.orbital_energies, n_occ_spatial=1)
    e_fci_tot, rdm1_fci, _ = run_2e_singlet_fci(h_mo, eri_mo, e_nuc)
    e_fci_corr = e_fci_tot - rhf.e_total

    assert e_mp2_corr < 0.0 and e_ccsd_corr < 0.0
    assert np.isclose(e_ccsd_corr, e_fci_corr, atol=1e-9)
    assert np.isclose(np.trace(rdm1_fci), 2.0, atol=1e-12)

    ext = compare_size_extensivity_dimers(e_mp2_corr, e_ccsd_corr, e_fci_corr, c0_monomer=0.985)
    assert ext["mp2_size_extensivity_error"] == 0.0
    assert ext["ccsd_size_extensivity_error"] == 0.0
    assert ext["cisd_size_extensivity_error"] > 0.0


def test_schmidt_decomposition_and_cumulant_energy_exactness() -> None:
    """Verify Schmidt decomposition bath count and exactness of the cumulant-based BE energy."""
    n_sites = 10
    H = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        H[i, (i + 1) % n_sites] = -1.0
        H[(i + 1) % n_sites, i] = -1.0
    _, C = np.linalg.eigh(H)
    C_occ = C[:, : n_sites // 2]
    P_oao = C_occ @ C_occ.T

    schmidt_c = construct_schmidt_bath_from_c_occ(C_occ, frag_indices=[0, 1], center_indices=[0, 1])
    schmidt_p = construct_schmidt_bath_from_1rdm(P_oao, frag_indices=[0, 1])
    assert schmidt_c.n_bath == 2 and schmidt_p["n_bath"] == 2
    np.testing.assert_allclose(schmidt_c.T_emb.T @ schmidt_c.T_emb, np.eye(4), atol=1e-12)

    S, T, V, eri, e_nuc = _heh_plus_sto3g_model()
    rhf = run_rhf(S, T, V, eri, e_nuc, n_elec=2)
    C_mo = rhf.mo_coeff
    h_lo = C_mo.T @ (T + V) @ C_mo
    V_lo = mo_eri_transform(eri, C_mo)

    be_full = run_bootstrap_embedding_cumulant(
        h_lo, V_lo, e_nuc=e_nuc, n_elec=2, fragments=[[0, 1]], centers=[[0, 1]]
    )
    e_fci_tot, _, _ = run_2e_singlet_fci(h_lo, V_lo, e_nuc)
    assert np.isclose(be_full.e_be_total, e_fci_tot, atol=1e-10)


def test_bootstrap_embedding_matching_loop_improves_accuracy() -> None:
    """Verify that the Newton-Raphson BE matching loop drives density mismatch to zero and improves E_corr."""
    n_lo = 6
    h_chain = np.diag([-0.6, 0.4, -0.4, 0.5, -0.5, 0.6])
    for i in range(n_lo - 1):
        h_chain[i, i + 1] = h_chain[i + 1, i] = -1.0
    V_chain = np.zeros((n_lo, n_lo, n_lo, n_lo))
    for i in range(n_lo):
        V_chain[i, i, i, i] = 2.0
        for j in range(n_lo):
            if i != j:
                V_chain[i, i, j, j] = 0.6 / abs(i - j)
                if abs(i - j) == 1:
                    V_chain[i, j, i, j] = V_chain[j, i, j, i] = V_chain[i, j, j, i] = 0.15

    e_fci_tot, _, _ = run_active_space_fci(h_chain, V_chain, n_elec=6)

    frags_be1 = [[i] for i in range(n_lo)]
    centers_be1 = [[i] for i in range(n_lo)]
    res_be1 = run_bootstrap_embedding_cumulant(
        h_chain, V_chain, 0.0, 6, frags_be1, centers_be1
    )

    frags_be2 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 4]]
    centers_be2 = [[0], [1], [2], [3], [4], [5]]
    res_be2_unmatched = run_bootstrap_embedding_matching_loop(
        h_chain, V_chain, 0.0, 6, frags_be2, centers_be2, match_densities=False
    )
    res_be2_matched = run_bootstrap_embedding_matching_loop(
        h_chain, V_chain, 0.0, 6, frags_be2, centers_be2, match_densities=True
    )

    # 1. Verify matching loop reduces RMS density mismatch from >1e-3 to <1e-9
    assert res_be2_unmatched.initial_rms_mismatch > 1e-3
    assert res_be2_matched.final_rms_mismatch < 1e-9
    assert np.isclose(res_be2_matched.total_center_electrons, 6.0, atol=1e-9)

    # 2. Verify systematic correlation energy accuracy improvement: BE1 -> BE2 (unmatched) -> BE2 (matched)
    e_corr_exact = e_fci_tot - res_be1.e_hf_total
    err_be1 = abs(res_be1.e_corr_cumulant - e_corr_exact)
    err_be2_unm = abs(res_be2_unmatched.e_corr_cumulant - e_corr_exact)
    err_be2_mat = abs(res_be2_matched.e_corr_cumulant - e_corr_exact)
    assert err_be2_unm < err_be1
    assert err_be2_mat < err_be2_unm