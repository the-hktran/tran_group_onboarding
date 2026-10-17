# Project 5 Instructions: Correlated Wavefunctions (CI, MP2, CCSD) & Bootstrap Embedding Matching Loops

## Objective
Implement and verify the hierarchy of correlated electronic structure methods (**MP2**, **CISD / Full CI**, and **spin-orbital CCSD**) alongside **Bootstrap Embedding ([BE](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01267))** with **Schmidt decomposition bath construction**, **Newton–Raphson edge-to-center density matching loops**, and the **cumulant-based correlation energy functional** ([QuEmb Toolbox](https://github.com/troyvvgroup/quemb)).

## Deliverables
1. **Part 5A (`correlated_methods_ci_mp2_ccsd.py`)**: $\mathcal{O}(N^5)$ MO integral transformation, MP2, Stanton–Gauss spin-orbital CCSD, 2e singlet FCI, and dimer size-extensivity comparisons.
2. **Part 5B (`schmidt_bath_construction.py`)**: Schmidt SVD of $C_{\text{occ}, \bar{\mathbb{F}}_A}$, Newton–Raphson BE density matching loop (`run_bootstrap_embedding_matching_loop`), and cumulant-based BE correlation energy (`compute_fragment_cumulant_energy`).
3. **Part 5C (`test_ci_mp2_ccsd_be.py`)**: Verify all functions via `pytest test_ci_mp2_ccsd_be.py`.
