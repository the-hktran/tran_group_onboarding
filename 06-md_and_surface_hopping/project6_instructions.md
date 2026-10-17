# Project 6 Instructions: Velocity Verlet NVE Dynamics & Tully's Fewest-Switches Surface Hopping

## Objective
Implement and test a symplectic [Velocity Verlet](https://docs.matlantis.com/atomistic-simulation-tutorial/en/6_1_md-nve.html) molecular dynamics integrator and Tully's [Fewest-Switches Surface Hopping (FSSH)](https://pubs.acs.org/acsodf/article/7/50/45810/461699/Pedagogical-Overview-of-the-Fewest-Switches) algorithm on a two-state avoided crossing (`fssh_tully_solver.py`).

## Verification
```bash
pytest test_md_fssh.py -v
```
