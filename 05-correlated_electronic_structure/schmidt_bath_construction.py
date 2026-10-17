"""Module 05B Project: Bootstrap Embedding — Schmidt Decomposition, Matching Loops & Cumulant Energy."""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.linalg import eigh, svd
from correlated_methods_ci_mp2_ccsd import run_2e_singlet_fci


@dataclass
class FragmentSchmidtSpace:
    frag_indices: list[int]
    center_indices_in_frag: list[int]  # Local indices in [0 .. n_frag-1] belonging to C_A
    edge_indices_in_frag: list[int]  # Local indices in [0 .. n_frag-1] belonging to E_A
    n_frag: int
    n_bath: int
    singular_values: np.ndarray
    T_emb: np.ndarray  # Shape (n_lo, n_frag + n_bath): transforms LO -> (FO + BO)
    P_env_lo: np.ndarray  # Spin-summed frozen environment 1-RDM in LO basis


@dataclass
class BootstrapEmbeddingResult:
    e_hf_total: float
    e_corr_cumulant: float
    e_be_total: float
    fragment_energies_corr: list[float]
    total_center_electrons: float
    initial_rms_mismatch: float = 0.0
    final_rms_mismatch: float = 0.0
    n_matching_iterations: int = 0
    chemical_potential_mu: float = 0.0
    edge_potentials_lambda: np.ndarray | None = None


def run_active_space_fci(
    h1e: np.ndarray, eri: np.ndarray, n_elec: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve exact spin-restricted Active-Space Full CI for n_orb <= 6 and return (E_elec, 1-RDM, 2-RDM).

    Works for 2-electron, 4-electron, or 6-electron active fragment+bath spaces so students can
    run multi-site overlapping BE(2) and BE(3) fragments in pure NumPy/SciPy.
    """
    if n_elec == 2 and h1e.shape[0] <= 4:
        return run_2e_singlet_fci(h1e, eri, 0.0)

    n_orb = h1e.shape[0]
    n_occ = n_elec // 2
    configs = list(itertools.combinations(range(n_orb), n_occ))
    n_cfg = len(configs)
    cfg_to_idx = {c: idx for idx, c in enumerate(configs)}

    # Build 1-particle excitation matrices <I | a^\dagger_p a_q | J> in single-spin sector
    E_1p = np.zeros((n_orb, n_orb, n_cfg, n_cfg), dtype=np.float64)
    for ket_idx, ket in enumerate(configs):
        ket_set = set(ket)
        for q in ket:
            for p in range(n_orb):
                if p == q:
                    E_1p[p, p, ket_idx, ket_idx] = 1.0
                elif p not in ket_set:
                    new_list = sorted((ket_set - {q}) | {p})
                    bra_idx = cfg_to_idx[tuple(new_list)]
                    pos_q = ket.index(q)
                    pos_p = new_list.index(p)
                    E_1p[p, q, bra_idx, ket_idx] = (-1.0) ** (pos_p + pos_q)

    h_mod = h1e - 0.5 * np.einsum("pqqs->ps", eri)
    H_mod_mat = np.einsum("pq,pqij->ij", h_mod, E_1p)
    I_cfg = np.eye(n_cfg)
    H_full = np.kron(H_mod_mat, I_cfg) + np.kron(I_cfg, H_mod_mat)

    V_same = 0.5 * np.einsum("pqrs,pqik,rskj->ij", eri, E_1p, E_1p, optimize=True)
    H_full += np.kron(V_same, I_cfg) + np.kron(I_cfg, V_same)

    E_flat = E_1p.reshape(n_orb * n_orb, n_cfg, n_cfg)
    V_mat = eri.reshape(n_orb * n_orb, n_orb * n_orb)
    evals_v, evecs_v = eigh(V_mat)
    for val, vec in zip(evals_v, evecs_v.T):
        if abs(val) > 1e-13:
            Op = np.einsum("x,xij->ij", vec, E_flat)
            H_full += val * np.kron(Op, Op)

    evals, evecs = eigh(H_full)
    e0 = float(evals[0])
    C = evecs[:, 0].reshape(n_cfg, n_cfg)

    EC = np.einsum("pqij,jk->pqik", E_1p, C)
    CE = np.einsum("ij,pqkj->pqik", C, E_1p)
    E_tot_C = EC + CE
    rdm1 = np.einsum("ij,pqij->pq", C, E_tot_C)
    EE_mat = np.einsum("qpij,rsij->pqrs", E_tot_C, E_tot_C, optimize=True)
    rdm2 = EE_mat - np.einsum("qr,ps->pqrs", np.eye(n_orb), rdm1)
    return e0, rdm1, rdm2


def construct_schmidt_bath_from_c_occ(
    C_occ_lo: np.ndarray,
    frag_indices: list[int],
    center_indices: list[int],
    svd_tol: float = 1e-7,
) -> FragmentSchmidtSpace:
    """Perform Schmidt Decomposition on C_occ in the LO basis to obtain FOs, BOs, and EOs."""
    n_lo = C_occ_lo.shape[0]
    comp_indices = [i for i in range(n_lo) if i not in frag_indices]
    center_in_frag = [frag_indices.index(c) for c in center_indices]
    edge_in_frag = [
        idx for idx in range(len(frag_indices)) if idx not in center_in_frag
    ]

    if len(comp_indices) == 0:
        T_emb = np.eye(n_lo, dtype=np.float64)
        return FragmentSchmidtSpace(
            frag_indices=frag_indices,
            center_indices_in_frag=center_in_frag,
            edge_indices_in_frag=edge_in_frag,
            n_frag=n_lo,
            n_bath=0,
            singular_values=np.array([], dtype=np.float64),
            T_emb=T_emb,
            P_env_lo=np.zeros((n_lo, n_lo), dtype=np.float64),
        )

    C_comp = C_occ_lo[comp_indices, :]
    U_comp, s_vals, _ = svd(C_comp, full_matrices=False)

    bath_mask = (s_vals > svd_tol) & (s_vals < 1.0 - svd_tol)
    env_mask = s_vals >= 1.0 - svd_tol

    U_bath = U_comp[:, bath_mask]
    U_env = U_comp[:, env_mask]
    n_frag = len(frag_indices)
    n_bath = U_bath.shape[1]

    T_emb = np.zeros((n_lo, n_frag + n_bath), dtype=np.float64)
    for col, lo_idx in enumerate(frag_indices):
        T_emb[lo_idx, col] = 1.0
    for col in range(n_bath):
        T_emb[comp_indices, n_frag + col] = U_bath[:, col]

    P_env_comp = (
        2.0 * (U_env @ U_env.T)
        if U_env.shape[1] > 0
        else np.zeros((len(comp_indices), len(comp_indices)))
    )
    P_env_lo = np.zeros((n_lo, n_lo), dtype=np.float64)
    P_env_lo[np.ix_(comp_indices, comp_indices)] = P_env_comp

    return FragmentSchmidtSpace(
        frag_indices=frag_indices,
        center_indices_in_frag=center_in_frag,
        edge_indices_in_frag=edge_in_frag,
        n_frag=n_frag,
        n_bath=n_bath,
        singular_values=s_vals[bath_mask],
        T_emb=T_emb,
        P_env_lo=P_env_lo,
    )


def construct_schmidt_bath_from_1rdm(
    P_oao: np.ndarray, frag_indices: list[int], svd_tol: float = 1e-8
) -> dict[str, np.ndarray | int]:
    """Extract entangled DMET/BE bath orbitals directly from an idempotent 1-RDM block P_{env, frag}."""
    n_ao = P_oao.shape[0]
    env_indices = [i for i in range(n_ao) if i not in frag_indices]
    P_env_frag = P_oao[np.ix_(env_indices, frag_indices)]

    U_env, singular_vals, _ = svd(P_env_frag, full_matrices=False)
    active_mask = singular_vals > svd_tol
    n_bath = int(np.sum(active_mask))
    U_bath = U_env[:, active_mask]

    n_frag = len(frag_indices)
    R_emb = np.zeros((n_ao, n_frag + n_bath), dtype=np.float64)
    for col, ao_idx in enumerate(frag_indices):
        R_emb[ao_idx, col] = 1.0
    for col in range(n_bath):
        R_emb[env_indices, n_frag + col] = U_bath[:, col]

    return {
        "singular_values": singular_vals,
        "n_frag": n_frag,
        "n_bath": n_bath,
        "R_emb": R_emb,
        "P_emb": R_emb.T @ P_oao @ R_emb,
    }


def compute_two_body_cumulant(
    rdm1_corr: np.ndarray,
    rdm2_corr: np.ndarray,
    rdm1_hf: np.ndarray,
    use_approximate_be_cumulant: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the correlated 1-RDM change dP = P - P_HF and two-body cumulant K_{pqrs}."""
    delta_p = rdm1_corr - rdm1_hf
    disc_corr = np.einsum("pq,rs->pqrs", rdm1_corr, rdm1_corr) - 0.5 * np.einsum(
        "ps,rq->pqrs", rdm1_corr, rdm1_corr
    )
    k_tilde = rdm2_corr - disc_corr

    if not use_approximate_be_cumulant:
        return delta_p, k_tilde

    delta_disc = np.einsum("pq,rs->pqrs", delta_p, delta_p) - 0.5 * np.einsum(
        "ps,rq->pqrs", delta_p, delta_p
    )
    k_approx = k_tilde + delta_disc
    return delta_p, k_approx


def compute_fragment_cumulant_energy(
    center_indices_in_frag: list[int],
    F_hf_emb: np.ndarray,
    V_emb: np.ndarray,
    delta_p: np.ndarray,
    k2_cumulant: np.ndarray,
) -> float:
    """Evaluate the cumulant-based correlation energy contribution of a single fragment A."""
    c_idx = center_indices_in_frag
    e_1body = float(np.einsum("pq,pq->", F_hf_emb[c_idx, :], delta_p[c_idx, :]))
    e_2body = 0.5 * float(
        np.einsum(
            "pqrs,pqrs->",
            V_emb[c_idx, :, :, :],
            k2_cumulant[c_idx, :, :, :],
            optimize=True,
        )
    )
    return e_1body + e_2body


def run_bootstrap_embedding_cumulant(
    h_lo: np.ndarray,
    V_lo: np.ndarray,
    e_nuc: float,
    n_elec: int,
    fragments: list[list[int]],
    centers: list[list[int]],
) -> BootstrapEmbeddingResult:
    """Run one-shot (unmatched) Bootstrap Embedding with Schmidt bath and cumulant energy."""
    return run_bootstrap_embedding_matching_loop(
        h_lo=h_lo,
        V_lo=V_lo,
        e_nuc=e_nuc,
        n_elec=n_elec,
        fragments=fragments,
        centers=centers,
        match_densities=False,
    )


def run_bootstrap_embedding_matching_loop(
    h_lo: np.ndarray,
    V_lo: np.ndarray,
    e_nuc: float,
    n_elec: int,
    fragments: list[list[int]],
    centers: list[list[int]],
    match_densities: bool = True,
    max_newton_iter: int = 15,
    rms_tol: float = 1e-9,
) -> BootstrapEmbeddingResult:
    """Run Bootstrap Embedding with Newton-Raphson edge-to-center density matching and global mu loop.

    For each overlapping fragment A:
      - Center sites C_A have global chemical potential mu added to h^A_{pp} (p in C_A).
      - Edge sites E_A have local Lagrange multiplier lambda^A_p added to h^A_{pp} (p in E_A)
        so that P^A_{pp}(Lambda) = P^B_{qq}(Lambda), where fragment B is the fragment whose
        center C_B contains that orbital!
      - Once L(Lambda*) = 0 converges, the cumulant-based energy is evaluated using the
        converged RDMs P^A(Lambda*) and Gamma^A(Lambda*) contracted with the UNDRESSED
        fragment Fock matrix F^{HF, A} and 2-electron integrals V^A_{pqrs} (Eq. 27 of LargeBasisBE).
    """
    from rhf_solver import run_rhf

    n_lo = h_lo.shape[0]
    rhf = run_rhf(np.eye(n_lo), h_lo, np.zeros_like(h_lo), V_lo, e_nuc, n_elec=n_elec)

    n_occ = n_elec // 2
    C_occ_lo = rhf.mo_coeff[:, :n_occ]
    P_hf_lo = 2.0 * (C_occ_lo @ C_occ_lo.T)
    J_hf = np.einsum("uvls,ls->uv", V_lo, P_hf_lo, optimize=True)
    K_hf = np.einsum("ulvs,ls->uv", V_lo, P_hf_lo, optimize=True)
    F_hf_lo = h_lo + J_hf - 0.5 * K_hf

    # Map each global LO index to the unique fragment B where that orbital is in C_B
    lo_to_center_frag: dict[int, tuple[int, int]] = {}
    for B, (f_idx, c_idx) in enumerate(zip(fragments, centers)):
        for c_lo in c_idx:
            lo_to_center_frag[c_lo] = (B, f_idx.index(c_lo))

    # Build Schmidt spaces and undressed fragment Hamiltonians
    frag_spaces: list[tuple[FragmentSchmidtSpace, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
    # Track all edge matching constraints: (frag_A, local_edge_p, target_frag_B, local_center_q)
    edge_constraints: list[tuple[int, int, int, int]] = []

    for A, (f_idx, c_idx) in enumerate(zip(fragments, centers)):
        sch = construct_schmidt_bath_from_c_occ(C_occ_lo, f_idx, c_idx)
        T = sch.T_emb
        J_env = np.einsum("uvls,ls->uv", V_lo, sch.P_env_lo, optimize=True)
        K_env = np.einsum("ulvs,ls->uv", V_lo, sch.P_env_lo, optimize=True)
        h_eff_lo = h_lo + J_env - 0.5 * K_env
        h_emb = T.T @ h_eff_lo @ T
        V_emb = np.einsum("uvls,up,vq,lr,st->pqrt", V_lo, T, T, T, T, optimize=True)
        F_hf_emb = T.T @ F_hf_lo @ T
        P_hf_emb = T.T @ P_hf_lo @ T
        n_elec_emb = int(round(float(np.trace(P_hf_emb))))
        frag_spaces.append((sch, h_emb, V_emb, F_hf_emb, P_hf_emb, n_elec_emb))

        for local_e in sch.edge_indices_in_frag:
            global_lo = f_idx[local_e]
            target_B, target_local_c = lo_to_center_frag[global_lo]
            edge_constraints.append((A, local_e, target_B, target_local_c))

    n_edge_params = len(edge_constraints)

    def solve_all_fragments(
        params: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        lambdas = params[:n_edge_params]
        mu = float(params[n_edge_params]) if len(params) > n_edge_params else 0.0

        # Group edge potentials by fragment A
        frag_edge_pots: dict[int, list[tuple[int, float]]] = {
            A: [] for A in range(len(fragments))
        }
        for k_idx, (A, local_e, _, _) in enumerate(edge_constraints):
            frag_edge_pots[A].append((local_e, float(lambdas[k_idx])))

        rdm1_list: list[np.ndarray] = []
        rdm2_list: list[np.ndarray] = []
        for A, (sch, h_emb, V_emb, _, _, n_elec_emb) in enumerate(frag_spaces):
            h_dressed = h_emb.copy()
            # Apply global chemical potential mu to center sites C_A
            for c_local in sch.center_indices_in_frag:
                h_dressed[c_local, c_local] += mu
            # Apply local Lagrange multipliers lambda^A to edge sites E_A
            for e_local, lam_val in frag_edge_pots[A]:
                h_dressed[e_local, e_local] += lam_val

            _, r1, r2 = run_active_space_fci(h_dressed, V_emb, n_elec=n_elec_emb)
            rdm1_list.append(r1)
            rdm2_list.append(r2)
        return rdm1_list, rdm2_list

    def evaluate_residuals(params: np.ndarray) -> np.ndarray:
        rdm1_list, _ = solve_all_fragments(params)
        res = np.zeros(n_edge_params + 1, dtype=np.float64)
        for k_idx, (A, local_e, B, local_c) in enumerate(edge_constraints):
            res[k_idx] = rdm1_list[A][local_e, local_e] - rdm1_list[B][local_c, local_c]
        total_ne = sum(
            sum(rdm1_list[A][c_loc, c_loc] for c_loc in frag_spaces[A][0].center_indices_in_frag)
            for A in range(len(fragments))
        )
        res[n_edge_params] = total_ne - n_elec
        return res

    params = np.zeros(n_edge_params + 1, dtype=np.float64)
    r_init = evaluate_residuals(params)
    init_rms = float(np.sqrt(np.mean(r_init**2)))
    curr_rms = init_rms
    n_iter_done = 0

    if match_densities and n_edge_params > 0:
        for it in range(1, max_newton_iter + 1):
            r_curr = evaluate_residuals(params)
            curr_rms = float(np.sqrt(np.mean(r_curr**2)))
            if curr_rms < rms_tol:
                break
            n_iter_done = it
            eps = 1e-5
            J_mat = np.zeros(( n_edge_params + 1, n_edge_params + 1), dtype=np.float64)
            for j in range(n_edge_params + 1):
                p_pert = params.copy()
                p_pert[j] += eps
                J_mat[:, j] = (evaluate_residuals(p_pert) - r_curr) / eps
            step, *_ = np.linalg.lstsq(J_mat, -r_curr, rcond=1e-10)
            params = params + step
        curr_rms = float(np.sqrt(np.mean(evaluate_residuals(params) ** 2)))

    rdm1_final, rdm2_final = solve_all_fragments(params)
    frag_corr_energies: list[float] = []
    total_center_nelec = 0.0

    for A, (sch, _, V_emb, F_hf_emb, P_hf_emb, _) in enumerate(frag_spaces):
        delta_p, k2_cumulant = compute_two_body_cumulant(
            rdm1_final[A], rdm2_final[A], P_hf_emb, use_approximate_be_cumulant=True
        )
        de_frag = compute_fragment_cumulant_energy(
            sch.center_indices_in_frag, F_hf_emb, V_emb, delta_p, k2_cumulant
        )
        frag_corr_energies.append(de_frag)
        total_center_nelec += float(
            sum(rdm1_final[A][c_loc, c_loc] for c_loc in sch.center_indices_in_frag)
        )

    e_corr_total = float(sum(frag_corr_energies))
    return BootstrapEmbeddingResult(
        e_hf_total=rhf.e_total,
        e_corr_cumulant=e_corr_total,
        e_be_total=rhf.e_total + e_corr_total,
        fragment_energies_corr=frag_corr_energies,
        total_center_electrons=total_center_nelec,
        initial_rms_mismatch=init_rms,
        final_rms_mismatch=curr_rms,
        n_matching_iterations=n_iter_done,
        chemical_potential_mu=float(params[n_edge_params]),
        edge_potentials_lambda=params[:n_edge_params].copy(),
    )


if __name__ == "__main__":
    # Compare BE1, Unmatched BE2, and Density-Matched BE2 on an inhomogeneous 6-site chain
    n_lo = 6
    h_chain = np.diag([-0.6, 0.4, -0.4, 0.5, -0.5, 0.6])
    for i in range(n_lo - 1):
        h_chain[i, i + 1] = h_chain[i + 1, i] = -1.0
    V_chain = np.zeros((n_lo, n_lo, n_lo, n_lo))
    for i in range(n_lo):
        V_chain[i, i, i, i] = 2.0
        for j in range(n_lo):
            if i != j:
                V_chain[i, i, j, j] = 0.6 / abs(i - j)
                if abs(i - j) == 1:
                    V_chain[i, j, i, j] = V_chain[j, i, j, i] = V_chain[i, j, j, i] = 0.15

    e_fci_tot, _, _ = run_active_space_fci(h_chain, V_chain, n_elec=6)
    frags_be1 = [[i] for i in range(n_lo)]
    centers_be1 = [[i] for i in range(n_lo)]
    res_be1 = run_bootstrap_embedding_cumulant(h_chain, V_chain, 0.0, 6, frags_be1, centers_be1)

    frags_be2 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 4]]
    centers_be2 = [[0], [1], [2], [3], [4], [5]]
    res_be2_unm = run_bootstrap_embedding_matching_loop(
        h_chain, V_chain, 0.0, 6, frags_be2, centers_be2, match_densities=False
    )
    res_be2_mat = run_bootstrap_embedding_matching_loop(
        h_chain, V_chain, 0.0, 6, frags_be2, centers_be2, match_densities=True
    )

    e_corr_ex = e_fci_tot - res_be1.e_hf_total
    print(f"Exact 6-site FCI E_corr:  {e_corr_ex:.6f} Eh")
    print(f"BE(1) E_corr:             {res_be1.e_corr_cumulant:.6f} Eh")
    print(f"BE(2) Unmatched E_corr:   {res_be2_unm.e_corr_cumulant:.6f} Eh (RMS mismatch = {res_be2_unm.initial_rms_mismatch:.2e})")
    print(
        f"BE(2) Matched E_corr:     {res_be2_mat.e_corr_cumulant:.6f} Eh "
        f"(RMS mismatch = {res_be2_mat.final_rms_mismatch:.2e} in {res_be2_mat.n_matching_iterations} Newton steps)"
    )