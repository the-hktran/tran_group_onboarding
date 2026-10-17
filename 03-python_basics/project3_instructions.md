# Project 3 Instructions: Vectorized Hückel Solver & Morse Potential Energy Surface Fitting

## Objective
Master loop-free [NumPy](https://numpy.org/neps/nep-0029-deprecation_policy.html) array indexing, Hermitian diagonalization (`scipy.linalg.eigh`), non-linear least-squares curve fitting (`scipy.optimize.curve_fit`), and automated unit testing with [Pytest](https://docs.pytest.org/en/latest/).

## Deliverables
1. **Part 3A (`huckel_solver.py`)**: Build Hückel Hamiltonians without Python loops and transform 4-index ERIs in $\mathcal{O}(N^5)$.
2. **Part 3B (`fit_morse_pes.py`)**: Fit diatomic dissociation curves to $V_{\text{Morse}}(R) = E_{\min} + D_e(1 - e^{-a(R-R_e)})^2$ and extract harmonic force constants $k = 2 D_e a^2$ and wavenumbers $\tilde{\nu}_e$ ($\text{cm}^{-1}$).
3. **Part 3C (`test_project3.py`)**: Verify all functions via `pytest test_project3.py`.
