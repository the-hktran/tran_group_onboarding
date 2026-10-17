"""Automated Pytest verification suite for Module 07 (MACE MD Pipeline & MCTDH Quantum Dynamics)."""
from pathlib import Path
import numpy as np
from mace_md_pipeline import (
    water_anharmonic_energy_and_forces,
    generate_md_training_snapshots,
    write_mace_extxyz,
    compute_committee_force_uncertainty,
)
from mctdh_2d_solver import run_2d_mctdh, run_exact_2d_dvr


def test_mace_forces_and_extxyz_generation(tmp_path: Path) -> None:
    """Verify analytical forces = -dE/dR and MACE extxyz dataset formatting."""
    snapshots = generate_md_training_snapshots(n_frames=5, seed=11)
    assert snapshots[0].config_type == "IsolatedAtom"
    assert snapshots[1].config_type == "IsolatedAtom"

    # Check analytical forces against finite difference on a water snapshot
    pos = snapshots[2].positions.copy()
    _, f_exact = water_anharmonic_energy_and_forces(pos)
    eps = 1e-6
    pos_plus = pos.copy()
    pos_plus[1, 0] += eps
    pos_minus = pos.copy()
    pos_minus[1, 0] -= eps
    e_plus, _ = water_anharmonic_energy_and_forces(pos_plus)
    e_minus, _ = water_anharmonic_energy_and_forces(pos_minus)
    f_num_x = -(e_plus - e_minus) / (2.0 * eps)
    assert np.isclose(f_exact[1, 0], f_num_x, atol=1e-6)

    xyz_file = tmp_path / "train.xyz"
    write_mace_extxyz(snapshots, xyz_file)
    text = xyz_file.read_text(encoding="utf-8")
    assert "REF_energy=" in text and "REF_forces:R:3" in text
    assert "config_type=IsolatedAtom" in text

    # Check active-learning committee uncertainty metric
    base_f = np.zeros((4, 3, 3))
    base_f[0, 0, 0] = 0.2
    base_f[1, 0, 0] = -0.2
    atom_sig, max_sig = compute_committee_force_uncertainty(base_f)
    assert max_sig > 0.0 and np.isclose(max_sig, atom_sig[0])


def test_mctdh_convergence_to_exact_dvr() -> None:
    """Verify that MCTDH (n_spf=4) conserves norm/energy and converges to exact 2D DVR dynamics."""
    c_exact = run_exact_2d_dvr(n_grid=18, dt=0.02, n_steps=80)
    res_tdh = run_2d_mctdh(n_spf=1, n_grid=18, dt=0.02, n_steps=80)
    res_mctdh = run_2d_mctdh(n_spf=4, n_grid=18, dt=0.02, n_steps=80)

    # Check norm and energy conservation of MCTDH
    assert np.max(np.abs(res_mctdh.norm_history - 1.0)) < 1e-5
    assert np.max(np.abs(res_mctdh.energy_history - res_mctdh.energy_history[0])) < 1e-4

    # Check that n_spf=4 MCTDH is dramatically more accurate than n_spf=1 TDH
    err_tdh = float(np.max(np.abs(res_tdh.autocorrelation - c_exact)))
    err_mctdh = float(np.max(np.abs(res_mctdh.autocorrelation - c_exact)))
    assert err_mctdh < 1e-3
    assert err_mctdh < 0.05 * err_tdh