# Project 8 Instructions: 1D Matrix Product State (TD-DMRG) Transport & Non-Markovian HEOM Spin-Boson Dynamics

## Objective
Implement and verify:
1. **1D Matrix Product State (MPS) / [TD-DMRG](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.8b00628) propagation (`tddmrg_mps_solver.py`)** via second-order Suzuki–Trotter two-site gates and SVD bond-dimension compression.
2. **Tanimura–Kubo Hierarchical Equations of Motion ([HEOM](https://qutip.org/docs/4.7/guide/guide-heom.html)) (`heom_spin_boson_solver.py`)** for a spin qubit coupled to a non-Markovian Drude–Lorentz phonon bath.

## Verification
```bash
pytest test_tddmrg_heom.py -v
```
