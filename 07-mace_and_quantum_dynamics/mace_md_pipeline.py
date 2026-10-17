"""Module 07A Project: Preparing Ab Initio MD Data for MACE Training & Active Learning."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class AtomicConfiguration:
    symbols: list[str]
    positions: np.ndarray  # Shape (N_atoms, 3) in Angstrom
    energy_ev: float  # Total potential energy in eV
    forces_ev_ang: np.ndarray  # Shape (N_atoms, 3) in eV / Angstrom
    config_type: str = "MD_Trajectory"


def water_anharmonic_energy_and_forces(
    positions: np.ndarray,
    e0_o: float = -2041.80,
    e0_h: float = -13.60,
    k_r: float = 36.5,
    r0: float = 0.9572,
    k_theta: float = 4.35,
    theta0_deg: float = 104.52,
) -> tuple[float, np.ndarray]:
    """Evaluate a smooth anharmonic H2O potential (eV) and analytical forces (eV/Angstrom).

    Atom ordering: [O, H1, H2].
    """
    r_o, r_h1, r_h2 = positions[0], positions[1], positions[2]
    v1 = r_h1 - r_o
    v2 = r_h2 - r_o
    d1 = float(np.linalg.norm(v1))
    d2 = float(np.linalg.norm(v2))
    u1 = v1 / d1
    u2 = v2 / d2

    cos_th = float(np.clip(np.dot(u1, u2), -0.999999, 0.999999))
    theta = float(np.arccos(cos_th))
    theta0 = float(np.deg2rad(theta0_deg))

    # Stretching + cubic anharmonicity + bend + stretch-stretch coupling
    dr1 = d1 - r0
    dr2 = d2 - r0
    dth = theta - theta0
    c_ss = 1.25  # stretch-stretch coupling (eV / Angstrom^2)

    e_bind = (
        0.5 * k_r * (dr1**2 + dr2**2)
        - 8.0 * (dr1**3 + dr2**3)
        + 0.5 * k_theta * dth**2
        + c_ss * dr1 * dr2
    )
    e_total = (e0_o + 2.0 * e0_h - 9.51) + e_bind

    # Analytical derivatives dE/dr1, dE/dr2, dE/dtheta
    de_dr1 = k_r * dr1 - 24.0 * dr1**2 + c_ss * dr2
    de_dr2 = k_r * dr2 - 24.0 * dr2**2 + c_ss * dr1
    de_dth = k_theta * dth

    sin_th = float(np.sqrt(max(1e-12, 1.0 - cos_th**2)))
    # d(theta)/dv1 = -1/sin(theta) * (u2 - cos(theta)*u1) / d1
    dth_dv1 = -(u2 - cos_th * u1) / (d1 * sin_th)
    dth_dv2 = -(u1 - cos_th * u2) / (d2 * sin_th)

    grad_h1 = de_dr1 * u1 + de_dth * dth_dv1
    grad_h2 = de_dr2 * u2 + de_dth * dth_dv2
    grad_o = -(grad_h1 + grad_h2)

    forces = -np.vstack([grad_o, grad_h1, grad_h2])
    return float(e_total), forces


def generate_md_training_snapshots(
    n_frames: int = 25, sigma_disp: float = 0.06, seed: int = 42
) -> list[AtomicConfiguration]:
    """Generate isolated atom references + rattled MD configurations for MACE training."""
    rng = np.random.default_rng(seed)
    configs: list[AtomicConfiguration] = [
        AtomicConfiguration(
            symbols=["O"],
            positions=np.zeros((1, 3)),
            energy_ev=-2041.80,
            forces_ev_ang=np.zeros((1, 3)),
            config_type="IsolatedAtom",
        ),
        AtomicConfiguration(
            symbols=["H"],
            positions=np.zeros((1, 3)),
            energy_ev=-13.60,
            forces_ev_ang=np.zeros((1, 3)),
            config_type="IsolatedAtom",
        ),
    ]

    r0 = 0.9572
    th0 = np.deg2rad(104.52)
    eq_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [r0 * np.sin(th0 / 2.0), 0.0, r0 * np.cos(th0 / 2.0)],
            [-r0 * np.sin(th0 / 2.0), 0.0, r0 * np.cos(th0 / 2.0)],
        ],
        dtype=np.float64,
    )

    for _ in range(n_frames):
        pos = eq_pos + rng.normal(0.0, sigma_disp, size=eq_pos.shape)
        e_val, f_val = water_anharmonic_energy_and_forces(pos)
        configs.append(
            AtomicConfiguration(
                symbols=["O", "H", "H"],
                positions=pos,
                energy_ev=e_val,
                forces_ev_ang=f_val,
                config_type="MD_300K",
            )
        )
    return configs


def write_mace_extxyz(configs: list[AtomicConfiguration], filepath: str | Path) -> None:
    """Write configurations in MACE-compatible Extended XYZ (extxyz) format."""
    lines: list[str] = []
    for cfg in configs:
        n_atoms = len(cfg.symbols)
        lines.append(f"{n_atoms}")
        header = (
            f'Properties=species:S:1:pos:R:3:REF_forces:R:3 '
            f'REF_energy={cfg.energy_ev:.8f} config_type={cfg.config_type} pbc="F F F"'
        )
        lines.append(header)
        for sym, pos, frc in zip(cfg.symbols, cfg.positions, cfg.forces_ev_ang):
            lines.append(
                f"{sym:2s} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f} "
                f"{frc[0]:14.8f} {frc[1]:14.8f} {frc[2]:14.8f}"
            )
    Path(filepath).write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_committee_force_uncertainty(
    committee_forces: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute per-atom and maximum committee force standard deviation sigma_F (eV/Ang).

    Parameters
    ----------
    committee_forces : np.ndarray
        Array of shape (M_models, N_atoms, 3) containing force predictions from M MACE models.
    """
    mean_forces = np.mean(committee_forces, axis=0, keepdims=True)
    sq_dev = np.sum((committee_forces - mean_forces) ** 2, axis=-1)  # (M_models, N_atoms)
    atom_sigma_f = np.sqrt(np.mean(sq_dev, axis=0))  # (N_atoms,)
    return atom_sigma_f, float(np.max(atom_sigma_f))


if __name__ == "__main__":
    all_configs = generate_md_training_snapshots(n_frames=20)
    write_mace_extxyz(all_configs[:16], "water_md_train.xyz")
    write_mace_extxyz(all_configs[2:8], "water_md_valid.xyz")
    print(f"Generated MACE extxyz files with {len(all_configs)} configurations.")