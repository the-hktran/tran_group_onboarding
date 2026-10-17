"""Module 08A Project: 1D Matrix Product State (MPS) & TD-DMRG / TEBD Exciton-Polaron Solver."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, svd


@dataclass
class TDDMRGResult:
    times: np.ndarray
    site_populations: np.ndarray  # Shape (n_steps+1, n_sites)
    mean_squared_displacement: np.ndarray  # Shape (n_steps+1,)
    entanglement_entropy: np.ndarray  # Half-chain von Neumann entropy S_vN(t)
    norm_history: np.ndarray


def initialize_product_mps(n_sites: int, excited_site: int) -> list[np.ndarray]:
    """Initialize a 1D MPS in a localized single-excitation state |0...010...0>.

    Each tensor A[k] has shape (chi_left, d=2, chi_right) with initial bond dimension chi = 1.
    Basis: s=0 -> empty site |0>, s=1 -> occupied exciton/polaron site |1>.
    """
    tensors: list[np.ndarray] = []
    for k in range(n_sites):
        A = np.zeros((1, 2, 1), dtype=np.complex128)
        if k == excited_site:
            A[0, 1, 0] = 1.0 + 0.0j
        else:
            A[0, 0, 0] = 1.0 + 0.0j
        tensors.append(A)
    return tensors


def mps_to_statevector(tensors: list[np.ndarray]) -> np.ndarray:
    """Contract an L-site MPS into a full 2^L statevector (for small-L verification)."""
    vec = tensors[0]  # (1, d, chi_1)
    for k in range(1, len(tensors)):
        # Contract rightmost virtual index of vec with leftmost virtual index of tensors[k]
        vec = np.tensordot(vec, tensors[k], axes=([-1], [0]))
    return vec.reshape(-1)


def apply_two_site_gate(
    A_left: np.ndarray,
    A_right: np.ndarray,
    gate_4d: np.ndarray,
    max_bond_dim: int = 16,
    svd_cutoff: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply a two-site gate U_{s1', s2', s1, s2} to adjacent MPS tensors and compress via SVD."""
    chi_l, d, _ = A_left.shape
    _, _, chi_r = A_right.shape

    # 1. Contract A_left (chi_l, s1, m) and A_right (m, s2, chi_r) -> Theta (chi_l, s1, s2, chi_r)
    theta = np.einsum("lsm,mtr->lstr", A_left, A_right, optimize=True)

    # 2. Apply two-site gate (s1', s2', s1, s2)
    theta_prime = np.einsum("abst,lstr->labr", gate_4d, theta, optimize=True)

    # 3. Reshape to (chi_l * d, d * chi_r) and perform SVD
    mat = theta_prime.reshape(chi_l * d, d * chi_r)
    U, S, Vh = svd(mat, full_matrices=False)

    # 4. Truncate bond dimension to max_bond_dim and svd_cutoff
    keep = S > svd_cutoff
    n_keep = max(1, min(int(np.sum(keep)), max_bond_dim))
    U = U[:, :n_keep]
    S = S[:n_keep]
    Vh = Vh[:n_keep, :]

    # Renormalize singular values to preserve norm and compute von Neumann entanglement entropy
    norm_sq = float(np.sum(S**2))
    S_norm = S / np.sqrt(norm_sq)
    p = S_norm**2
    p_pos = p[p > 1e-15]
    entropy = float(-np.sum(p_pos * np.log(p_pos)))

    # 5. Split sqrt(S) symmetrically into new left and right tensors
    sqrt_S = np.sqrt(S)
    new_left = (U * sqrt_S[None, :]).reshape(chi_l, d, n_keep)
    new_right = (sqrt_S[:, None] * Vh).reshape(n_keep, d, chi_r)
    return new_left, new_right, entropy


def build_two_site_xxz_gate(
    j_hop: float, v_int: float, de_left: float, de_right: float, dt: float
) -> np.ndarray:
    """Build 2-site unitary exp(-i * h_{k, k+1} * dt) in basis {|00>, |01>, |10>, |11>}."""
    h_mat = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, de_right, -j_hop, 0.0],
            [0.0, -j_hop, de_left, 0.0],
            [0.0, 0.0, 0.0, de_left + de_right + v_int],
        ],
        dtype=np.complex128,
    )
    U_mat = expm(-1j * h_mat * dt)
    return U_mat.reshape(2, 2, 2, 2)


def measure_site_populations(tensors: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Compute <n_k> on every site and total norm using efficient MPS transfer matrices."""
    n_sites = len(tensors)
    pops = np.zeros(n_sites, dtype=np.float64)
    n_op = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    for target in range(n_sites):
        env = np.ones((1, 1), dtype=np.complex128)
        for k, A in enumerate(tensors):
            op = n_op if k == target else np.eye(2, dtype=np.complex128)
            # Contract env_{a, a'} with A^*_{a, s, b}, op_{s, s'}, A_{a', s', b'}
            env = np.einsum("ac,asb,st,ctd->bd", env, A.conj(), op, A, optimize=True)
        pops[target] = float(np.real(env[0, 0]))

    # Compute total norm <Psi|Psi>
    env_norm = np.ones((1, 1), dtype=np.complex128)
    for A in tensors:
        env_norm = np.einsum("ac,asb,csd->bd", env_norm, A.conj(), A, optimize=True)
    norm_val = float(np.sqrt(np.real(env_norm[0, 0])))
    return pops, norm_val


def run_tddmrg_transport(
    n_sites: int = 7,
    excited_site: int = 3,
    j_hop: float = 1.0,
    v_int: float = 0.4,
    dt: float = 0.05,
    n_steps: int = 60,
    max_bond_dim: int = 16,
) -> TDDMRGResult:
    """Propagate a 1D chain via second-order Suzuki-Trotter TD-DMRG / TEBD sweeps."""
    tensors = initialize_product_mps(n_sites, excited_site)
    gate_half = build_two_site_xxz_gate(j_hop, v_int, 0.0, 0.0, 0.5 * dt)
    gate_full = build_two_site_xxz_gate(j_hop, v_int, 0.0, 0.0, dt)

    times = np.zeros(n_steps + 1)
    pops = np.zeros((n_steps + 1, n_sites))
    msd = np.zeros(n_steps + 1)
    entropies = np.zeros(n_steps + 1)
    norms = np.zeros(n_steps + 1)

    sites_idx = np.arange(n_sites, dtype=np.float64)
    p0, n0 = measure_site_populations(tensors)
    pops[0] = p0
    msd[0] = float(np.sum(((sites_idx - excited_site) ** 2) * p0))
    norms[0] = n0
    mid_bond = n_sites // 2 - 1

    for step in range(1, n_steps + 1):
        s_mid = 0.0
        # 1. Half-step on odd bonds (0, 2, 4, ...)
        for k in range(0, n_sites - 1, 2):
            tensors[k], tensors[k + 1], s_val = apply_two_site_gate(
                tensors[k], tensors[k + 1], gate_half, max_bond_dim=max_bond_dim
            )
            if k == mid_bond:
                s_mid = s_val

        # 2. Full-step on even bonds (1, 3, 5, ...)
        for k in range(1, n_sites - 1, 2):
            tensors[k], tensors[k + 1], s_val = apply_two_site_gate(
                tensors[k], tensors[k + 1], gate_full, max_bond_dim=max_bond_dim
            )
            if k == mid_bond:
                s_mid = s_val

        # 3. Half-step on odd bonds (0, 2, 4, ...)
        for k in range(0, n_sites - 1, 2):
            tensors[k], tensors[k + 1], s_val = apply_two_site_gate(
                tensors[k], tensors[k + 1], gate_half, max_bond_dim=max_bond_dim
            )
            if k == mid_bond:
                s_mid = s_val

        times[step] = step * dt
        p_curr, n_curr = measure_site_populations(tensors)
        pops[step] = p_curr
        msd[step] = float(np.sum(((sites_idx - excited_site) ** 2) * p_curr))
        entropies[step] = s_mid
        norms[step] = n_curr

    return TDDMRGResult(
        times=times,
        site_populations=pops,
        mean_squared_displacement=msd,
        entanglement_entropy=entropies,
        norm_history=norms,
    )


if __name__ == "__main__":
    res = run_tddmrg_transport(n_sites=7, excited_site=3, dt=0.05, n_steps=50)
    print(f"Final norm: {res.norm_history[-1]:.8f}")
    print(f"Final Mean-Squared Displacement (MSD): {res.mean_squared_displacement[-1]:.4f} sites^2")
    print("Final site populations:", np.round(res.site_populations[-1], 4))