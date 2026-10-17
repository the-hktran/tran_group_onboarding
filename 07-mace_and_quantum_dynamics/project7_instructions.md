# Project 7 Instructions: MACE Active-Learning Pipeline & 2D MCTDH Quantum Wavepacket Dynamics

## Objective
Build an end-to-end Extended XYZ (`extxyz`) training dataset generator and query-by-committee uncertainty filter for [MACE](https://mace-docs.readthedocs.io/en/latest/guide/training.html) (`mace_md_pipeline.py`), and propagate a 2D coupled anharmonic quantum wavepacket using **Multi-Configuration Time-Dependent Hartree ([MCTDH](https://pubs.aip.org/aip/jcp/article/154/11/114107/315442/A-rectangular-collocation-multi-configuration-time))** (`mctdh_2d_solver.py`).

## Verification
```bash
pytest test_mace_mctdh.py -v
```
