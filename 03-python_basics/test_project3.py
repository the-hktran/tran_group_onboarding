"""Automated Pytest suite for Module 03 (Hückel MO Solver, ERI Transform & Morse PES Fit)."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from huckel_solver import (
    build_huckel_hamiltonian,
    solve_pi_system,
    transform_eri_ao_to_mo,
)

try:
    from fit_morse_pes import morse_potential, fit_morse_curve
except ModuleNotFoundError:
    _spec = importlib.util.spec_from_file_location(
        "fit_morse_pes", Path(__file__).with_name("fit_morse_pes.py")
    )
    _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    morse_potential = _mod.morse_potential
    fit_morse_curve = _mod.fit_morse_curve


def test_huckel_linear_vs_cyclic_delocalization() -> None:
    """Verify Hückel eigenvalues for ethylene (N=2), butadiene (N=4), and benzene (N=6)."""
    H_eth = build_huckel_hamiltonian(2, alpha=0.0, beta=-1.0, cyclic=False)
    res_eth = solve_pi_system(H_eth, n_electrons=2)
    assert np.isclose(res_eth["total_pi_energy"], -2.0, atol=1e-12)
    assert np.isclose(res_eth["homo_lumo_gap"], 2.0, atol=1e-12)

    H_but = build_huckel_hamiltonian(4, alpha=0.0, beta=-1.0, cyclic=False)
    res_but = solve_pi_system(H_but, n_electrons=4)
    assert np.isclose(res_but["total_pi_energy"], -2.0 * np.sqrt(5.0), atol=1e-10)

    H_hex = build_huckel_hamiltonian(6, alpha=0.0, beta=-1.0, cyclic=False)
    H_ben = build_huckel_hamiltonian(6, alpha=0.0, beta=-1.0, cyclic=True)
    res_hex = solve_pi_system(H_hex, n_electrons=6)
    res_ben = solve_pi_system(H_ben, n_electrons=6)
    assert res_ben["total_pi_energy"] < res_hex["total_pi_energy"]
    assert np.isclose(res_ben["total_pi_energy"], -8.0, atol=1e-12)


def test_eri_ao_to_mo_orthogonal_invariance() -> None:
    """Verify that 4-index AO-to-MO integral transformation preserves total norm under unitary C."""
    rng = np.random.default_rng(123)
    n_ao = 5
    eri_ao = rng.normal(size=(n_ao, n_ao, n_ao, n_ao))
    C, _ = np.linalg.qr(rng.normal(size=(n_ao, n_ao)))
    eri_mo = transform_eri_ao_to_mo(eri_ao, C)
    assert np.isclose(np.linalg.norm(eri_ao), np.linalg.norm(eri_mo), atol=1e-10)


def test_morse_pes_parameter_recovery() -> None:
    """Verify that fit_morse_curve recovers exact Morse parameters and H2 harmonic frequency."""
    r_grid = np.linspace(0.85, 4.0, 30)
    d_e_true, a_true, r_eq_true, e_min_true = 0.1745, 1.0282, 1.4011, -1.1745
    energies = morse_potential(r_grid, d_e_true, a_true, r_eq_true, e_min_true)

    fit = fit_morse_curve(r_grid, energies, reduced_mass_amu=0.50391)
    assert np.isclose(fit.d_e_hartree, d_e_true, atol=1e-7)
    assert np.isclose(fit.a_inv_bohr, a_true, atol=1e-7)
    assert np.isclose(fit.r_eq_bohr, r_eq_true, atol=1e-7)
    assert np.isclose(fit.e_min_hartree, e_min_true, atol=1e-7)
    assert 4350.0 < fit.omega_e_cm1 < 4450.0
    assert fit.rmse_hartree < 1e-10