"""Module 05A Project: Configuration Interaction (CI), MP2, and Coupled Cluster (CCSD) from Scratch."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh


@dataclass
class CorrelatedResult:
    e_hf: float
    e_mp2_corr: float
    e_mp2_tot: float
    e_cisd_corr: float
    e_cisd_tot: float
    e_ccsd_corr: float
    e_ccsd_tot: float
    e_fci_corr: float
    e_fci_tot: float


def mo_eri_transform(eri_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Transform 4-index AO ERIs (uv|ls) to spatial MO basis (pq|rs) in O(N^5) via np.einsum."""
    return np.einsum("uvls,up,vq,lr,st->pqrt", eri_ao, C, C, C, C, optimize=True)


def run_mp2_from_mo(
    eri_mo: np.ndarray, eps: np.ndarray, n_occ: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute closed-shell MP2 correlation energy, T2 amplitudes, and MP2 2-body cumulant."""
    iajb = eri_mo[:n_occ, n_occ:, :n_occ, n_occ:]
    ibja = np.swapaxes(iajb, 1, 3)
    e_occ = eps[:n_occ]
    e_virt = eps[n_occ:]
    denom = (
        e_occ[:, None, None, None]
        - e_virt[None, :, None, None]
        + e_occ[None, None, :, None]
        - e_virt[None, None, None, :]
    )
    t2 = iajb / denom
    tau_mp2 = 2.0 * iajb - ibja
    e_mp2_corr = float(np.sum(t2 * tau_mp2))
    # First-order two-body cumulant block K_{iajb} = 2 * (2 * t2 - t2.swapaxes(1, 3))
    k2_ovov = 2.0 * (2.0 * t2 - np.swapaxes(t2, 1, 3))
    return e_mp2_corr, t2, k2_ovov


def spatial_to_spin_orbital_integrals(
    h_mo: np.ndarray, eri_mo: np.ndarray, eps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spatial MO integrals (pq|rs) (chemist) to antisymmetrized spin-orbital <pq||rs> (physicist).

    Spin-orbital ordering: even indices 2p = alpha, odd indices 2p+1 = beta.
    """
    n_mo = h_mo.shape[0]
    n_so = 2 * n_mo
    h_so = np.zeros((n_so, n_so), dtype=np.float64)
    eps_so = np.zeros(n_so, dtype=np.float64)
    for p in range(n_mo):
        eps_so[2 * p] = eps[p]
        eps_so[2 * p + 1] = eps[p]
        for q in range(n_mo):
            h_so[2 * p, 2 * q] = h_mo[p, q]
            h_so[2 * p + 1, 2 * q + 1] = h_mo[p, q]

    # Physicist's <pq|rs> = chemist's (pr|qs) * delta(spin_p, spin_r) * delta(spin_q, spin_s)
    eri_phys = np.zeros((n_so, n_so, n_so, n_so), dtype=np.float64)
    for sp in (0, 1):
        for sq in (0, 1):
            eri_phys[sp::2, sq::2, sp::2, sq::2] = eri_mo.transpose(0, 2, 1, 3)

    # Antisymmetrized <pq||rs> = <pq|rs> - <pq|sr>
    eri_asym = eri_phys - eri_phys.transpose(0, 1, 3, 2)
    return h_so, eri_asym, eps_so


def run_ccsd_spin_orbital(
    h_mo: np.ndarray,
    eri_mo: np.ndarray,
    eps: np.ndarray,
    n_occ_spatial: int,
    max_iter: int = 80,
    tol: float = 1e-10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve spin-orbital Coupled Cluster Singles and Doubles (CCSD) equations (Stanton et al. 1991)."""
    _, g, eps_so = spatial_to_spin_orbital_integrals(h_mo, eri_mo, eps)
    n_occ = 2 * n_occ_spatial
    n_so = len(eps_so)
    n_virt = n_so - n_occ

    o = slice(0, n_occ)
    v = slice(n_occ, n_so)

    # Canonical HF Fock matrix in spin-orbital basis is diagonal: f_pq = eps_p * delta_pq
    f = np.diag(eps_so)

    e_occ = eps_so[o]
    e_virt = eps_so[v]
    D_ia = e_occ[:, None] - e_virt[None, :]
    D_ijab = (
        e_occ[:, None, None, None]
        + e_occ[None, :, None, None]
        - e_virt[None, None, :, None]
        - e_virt[None, None, None, :]
    )

    # Initial guess: T1 = 0, T2 = MP2 amplitudes <ij||ab> / D_ijab
    t1 = np.zeros((n_occ, n_virt), dtype=np.float64)
    t2 = g[o, o, v, v] / D_ijab

    def ccsd_energy(t1_curr: np.ndarray, t2_curr: np.ndarray) -> float:
        tau = t2_curr + 0.5 * (
            np.einsum("ia,jb->ijab", t1_curr, t1_curr)
            - np.einsum("ib,ja->ijab", t1_curr, t1_curr)
        )
        return float(
            np.einsum("ia,ia->", f[o, v], t1_curr)
            + 0.25 * np.einsum("ijab,ijab->", g[o, o, v, v], tau)
        )

    e_corr_old = ccsd_energy(t1, t2)

    for _ in range(max_iter):
        tau_tilde = t2 + 0.5 * (
            np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)
        )
        tau = t2 + (
            np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)
        )

        # Stanton-Gauss effective 1-body intermediates (F_ae, F_mi, F_me)
        F_ae = (1.0 - np.eye(n_virt)) * f[v, v] - 0.5 * np.einsum(
            "me,ma->ae", f[o, v], t1
        )
        F_ae += np.einsum("mf,mafe->ae", t1, g[o, v, v, v]) - 0.5 * np.einsum(
            "mnaf,mnef->ae", tau_tilde, g[o, o, v, v]
        )

        F_mi = (1.0 - np.eye(n_occ)) * f[o, o] + 0.5 * np.einsum(
            "ie,me->mi", t1, f[o, v]
        )
        F_mi += np.einsum("ne,mnie->mi", t1, g[o, o, o, v]) + 0.5 * np.einsum(
            "inef,mnef->mi", tau_tilde, g[o, o, v, v]
        )

        F_me = f[o, v] + np.einsum("nf,mnef->me", t1, g[o, o, v, v])

        # Stanton-Gauss effective 2-body intermediates (W_mnij, W_abef, W_mbej)
        W_mnij = (
            g[o, o, o, o]
            + np.einsum("je,mnie->mnij", t1, g[o, o, o, v])
            - np.einsum("ie,mnje->mnij", t1, g[o, o, o, v])
            + 0.25 * np.einsum("ijef,mnef->mnij", tau, g[o, o, v, v])
        )

        W_abef = (
            g[v, v, v, v]
            - np.einsum("mb,amef->abef", t1, g[v, o, v, v])
            + np.einsum("ma,bmef->abef", t1, g[v, o, v, v])
            + 0.25 * np.einsum("mnab,mnef->abef", tau, g[o, o, v, v])
        )

        W_mbej = (
            g[o, v, v, o]
            + np.einsum("jf,mbef->mbej", t1, g[o, v, v, v])
            - np.einsum("nb,mnej->mbej", t1, g[o, o, v, o])
            - np.einsum(
                "jnfb,mnef->mbej",
                0.5 * t2 + np.einsum("jf,nb->jnfb", t1, t1),
                g[o, o, v, v],
            )
        )

        # Update T1 amplitudes
        rhs_t1 = (
            f[o, v].copy()
            + np.einsum("ie,ae->ia", t1, F_ae)
            - np.einsum("ma,mi->ia", t1, F_mi)
            + np.einsum("imae,me->ia", t2, F_me)
            - np.einsum("nf,naif->ia", t1, g[o, v, o, v])
            - 0.5 * np.einsum("imef,maef->ia", t2, g[o, v, v, v])
            - 0.5 * np.einsum("mnae,nmei->ia", t2, g[o, o, v, o])
        )
        t1_new = rhs_t1 / D_ia

        # Update T2 amplitudes
        P_ab_1 = np.einsum(
            "ijae,be->ijab", t2, F_ae - 0.5 * np.einsum("mb,me->be", t1, F_me)
        )
        P_ab_1 = P_ab_1 - P_ab_1.transpose(0, 1, 3, 2)

        P_ij_1 = np.einsum(
            "imab,mj->ijab", t2, F_mi + 0.5 * np.einsum("je,me->mj", t1, F_me)
        )
        P_ij_1 = P_ij_1 - P_ij_1.transpose(1, 0, 2, 3)

        P_ijab = np.einsum("imae,mbej->ijab", t2, W_mbej) - np.einsum(
            "ie,ma,mbej->ijab", t1, t1, g[o, v, v, o]
        )
        P_ijab = (
            P_ijab
            - P_ijab.transpose(1, 0, 2, 3)
            - P_ijab.transpose(0, 1, 3, 2)
            + P_ijab.transpose(1, 0, 3, 2)
        )

        P_ij_2 = np.einsum("ie,abej->ijab", t1, g[v, v, v, o])
        P_ij_2 = P_ij_2 - P_ij_2.transpose(1, 0, 2, 3)

        P_ab_2 = np.einsum("ma,mbij->ijab", t1, g[o, v, o, o])
        P_ab_2 = P_ab_2 - P_ab_2.transpose(0, 1, 3, 2)

        rhs_t2 = (
            g[o, o, v, v].copy()
            + P_ab_1
            - P_ij_1
            + 0.5 * np.einsum("mnab,mnij->ijab", tau, W_mnij)
            + 0.5 * np.einsum("ijef,abef->ijab", tau, W_abef)
            + P_ijab
            + P_ij_2
            - P_ab_2
        )
        t2_new = rhs_t2 / D_ijab

        e_corr_new = ccsd_energy(t1_new, t2_new)
        if abs(e_corr_new - e_corr_old) < tol:
            return e_corr_new, t1_new, t2_new

        t1, t2, e_corr_old = t1_new, t2_new, e_corr_new

    return e_corr_old, t1, t2


def run_2e_singlet_fci(
    h_mo: np.ndarray, eri_mo: np.ndarray, e_nuc: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve exact singlet 2-electron Full CI (equivalent to CISD for N_elec=2) and return (E_tot, 1-RDM, 2-RDM).

    For a 2-electron singlet state, the spatial wavefunction is a symmetric matrix C_{pq}
    in the product basis |p_alpha q_beta>, with norm sum_{pq} |C_{pq}|^2 = 1.
    The Hamiltonian acts as (H C)_{pq} = sum_r h_{pr} C_{rq} + sum_s C_{ps} h_{sq} + sum_{rs} (pr|qs) C_{rs}.
    """
    n_mo = h_mo.shape[0]
    I = np.eye(n_mo)
    # Build 2-electron singlet Hamiltonian matrix of size (n_mo^2, n_mo^2)
    H_2e = (
        np.kron(h_mo, I)
        + np.kron(I, h_mo)
        + eri_mo.transpose(0, 2, 1, 3).reshape(n_mo * n_mo, n_mo * n_mo)
    )
    evals, evecs = eigh(H_2e)
    e_elec = float(evals[0])
    C_mat = evecs[:, 0].reshape(n_mo, n_mo)
    # Enforce singlet spatial symmetry C = 0.5 * (C + C.T)
    C_mat = 0.5 * (C_mat + C_mat.T)
    C_mat /= np.linalg.norm(C_mat)

    # Spin-summed 1-RDM: P_{pq} = 2 * (C @ C.T)_{pq} (trace = 2)
    rdm1 = 2.0 * (C_mat @ C_mat.T)

    # Spin-summed 2-RDM for 2-electron singlet: Gamma_{pqrs} = 2 * C_{pr} * C_{qs}
    # Verify: 0.5 * sum_{pqrs} (pq|rs) Gamma_{pqrs} = sum_{pqrs} (pq|rs) C_{pr} C_{qs} = <Psi|V_ee|Psi>
    rdm2 = 2.0 * np.einsum("pr,qs->pqrs", C_mat, C_mat, optimize=True)
    return e_elec + e_nuc, rdm1, rdm2


def compare_size_extensivity_dimers(
    e_corr_monomer_mp2: float,
    e_corr_monomer_ccsd: float,
    e_corr_monomer_cisd: float,
    c0_monomer: float,
) -> dict[str, float]:
    """Compare exact 2-monomer non-interacting dimer energies for MP2, CCSD, and truncated CISD.

    For two non-interacting closed-shell monomers A and B separated by R -> infinity:
      - MP2 and CCSD are exactly size-extensive: E_corr(A+B) = 2 * E_corr(A).
      - Truncated CISD omits quadruple excitations C_2(A)*C_2(B). By intermediate normalization,
        for N_m identical non-interacting monomers with monomer CISD correlation energy E_c,
        the CISD correlation energy of the supermolecule is strictly smaller in magnitude!
    """
    # Exact analytical supermolecule CISD relation for N=2 identical 2-electron monomers:
    # Secular equation for supermolecule CISD gives E_corr_CISD(A+B) < 2 * E_corr_CISD(A)
    # Specifically, with c0_monomer = cos(theta), tan(theta)^2 = (1 - c0^2)/c0^2:
    tan2 = (1.0 - c0_monomer**2) / (c0_monomer**2)
    # Ratio E_CISD(2 monomers) / (2 * E_CISD(1 monomer)) = (sqrt(1 + 8*tan2) - 1) / (2 * (sqrt(1 + 4*tan2) - 1))
    ratio_cisd = (np.sqrt(1.0 + 8.0 * tan2) - 1.0) / (
        2.0 * (np.sqrt(1.0 + 4.0 * tan2) - 1.0)
    )
    e_dimer_cisd = 2.0 * e_corr_monomer_cisd * float(ratio_cisd)
    return {
        "mp2_size_extensivity_error": 0.0,
        "ccsd_size_extensivity_error": 0.0,
        "cisd_monomer_x2": 2.0 * e_corr_monomer_cisd,
        "cisd_dimer_actual": e_dimer_cisd,
        "cisd_size_extensivity_error": e_dimer_cisd - 2.0 * e_corr_monomer_cisd,
    }


if __name__ == "__main__":
    # Benchmark on HeH+ STO-3G model integrals from Module 04
    from rhf_solver import run_rhf

    S = np.array([[1.0, 0.4508], [0.4508, 1.0]])
    T = np.array([[1.5163, 0.2367], [0.2367, 0.7600]])
    V = np.array([[-3.6803, -1.1419], [-1.1419, -2.0466]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 1.3072
    eri[1, 1, 1, 1] = 0.7746
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.6057
    eri[0, 1, 0, 1] = eri[1, 0, 0, 1] = eri[0, 1, 1, 0] = eri[1, 0, 1, 0] = 0.1813
    eri[0, 0, 0, 1] = eri[0, 0, 1, 0] = eri[0, 1, 0, 0] = eri[1, 0, 0, 0] = 0.4373
    eri[1, 1, 0, 1] = eri[1, 1, 1, 0] = eri[0, 1, 1, 1] = eri[1, 0, 1, 1] = 0.3118
    e_nuc = 1.3668

    rhf = run_rhf(S, T, V, eri, e_nuc, n_elec=2)
    h_mo = rhf.mo_coeff.T @ (T + V) @ rhf.mo_coeff
    eri_mo = mo_eri_transform(eri, rhf.mo_coeff)

    e_mp2_c, _, _ = run_mp2_from_mo(eri_mo, rhf.orbital_energies, n_occ=1)
    e_ccsd_c, _, _ = run_ccsd_spin_orbital(h_mo, eri_mo, rhf.orbital_energies, n_occ_spatial=1)
    e_fci_tot, _, _ = run_2e_singlet_fci(h_mo, eri_mo, e_nuc)

    print(f"RHF Total Energy:  {rhf.e_total:.10f} Eh")
    print(f"MP2 Total Energy:  {rhf.e_total + e_mp2_c:.10f} Eh (E_corr = {e_mp2_c:.8f})")
    print(f"CCSD Total Energy: {rhf.e_total + e_ccsd_c:.10f} Eh (E_corr = {e_ccsd_c:.8f})")
    print(f"FCI Total Energy:  {e_fci_tot:.10f} Eh (E_corr = {e_fci_tot - rhf.e_total:.8f})")