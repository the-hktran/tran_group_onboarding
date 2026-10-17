"""Module 05 Example 01: Single-Point RHF Calculation & Mulliken vs. Löwdin Population Analysis."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh
from rhf_solver import run_rhf

try:
    from integrals import load_system_integrals
except ModuleNotFoundError:
    _spec = importlib.util.spec_from_file_location(
        "integrals", Path(__file__).with_name("integrals.py")
    )
    _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    load_system_integrals = _mod.load_system_integrals


def compute_mulliken_and_lowdin_charges(
    D_occ: np.ndarray,
    S: np.ndarray,
    ao_to_atom: list[int],
    nuclear_charges: list[float],
) -> dict[str, np.ndarray]:
    """Compute Mulliken (PS) and Löwdin (S^(1/2) P S^(1/2)) atomic partial charges."""
    P = 2.0 * D_occ
    ps_diag = np.diag(P @ S)

    evals, evecs = eigh(S)
    S_half = (evecs * np.sqrt(evals)) @ evecs.T
    p_lowdin_diag = np.diag(S_half @ P @ S_half)

    q_mulliken = np.array(nuclear_charges, dtype=np.float64)
    q_lowdin = np.array(nuclear_charges, dtype=np.float64)

    for mu, atom_idx in enumerate(ao_to_atom):
        q_mulliken[atom_idx] -= ps_diag[mu]
        q_lowdin[atom_idx] -= p_lowdin_diag[mu]

    return {
        "mulliken_ao_pop": ps_diag,
        "lowdin_ao_pop": p_lowdin_diag,
        "mulliken_charges": q_mulliken,
        "lowdin_charges": q_lowdin,
    }


if __name__ == "__main__":
    S, T, V, eri, e_nuc, n_elec = load_system_integrals("h2o_sto3g")
    rhf = run_rhf(S, T, V, eri, e_nuc, n_elec=n_elec)
    pops = compute_mulliken_and_lowdin_charges(
        rhf.density_matrix,
        S,
        ao_to_atom=[0, 0, 0, 0, 0, 1, 2],
        nuclear_charges=[8.0, 1.0, 1.0],
    )
    print(f"H2O RHF Total Energy:   {rhf.e_total:.8f} Eh ({rhf.n_iterations} iterations)")
    print("Mulliken charges [O, H1, H2]:", np.round(pops["mulliken_charges"], 4))
    print("Löwdin charges   [O, H1, H2]:", np.round(pops["lowdin_charges"], 4))