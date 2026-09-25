"""Core ANNNI physics for the Q-SITE 2026 Scientific Track (team module).

Conventions (see team/literature/WHAT_THE_LITERATURE_SAYS.md, [S28], [S23]):
  H = -sum_i Z_i Z_{i+1} + kappa sum_i Z_i Z_{i+2} - h sum_i X_i
  Pauli operators, J1 = 1, lattice spacing 1, periodic boundaries by default
  (N NN and N NNN bonds for N >= 5).
  Basis: PennyLane wire ordering, wire 0 is the most significant bit;
  Z eigenvalue of wire i in basis state s is 1 - 2*((s >> (N-1-i)) & 1).
  C(r)  = (1/N) sum_i <Z_i Z_{i+r}>          (C(0) = 1 kept)
  S(q)  = (1/N) sum_ij exp[iq(i-j)] <Z_i Z_j>  (self terms kept), m^2(q) = S(q)/N
  m_x   = (1/N) sum_i <X_i>
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ----------------------------------------------------------------------------
# basis helpers
# ----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def zsigns(n: int) -> np.ndarray:
    """(n, 2^n) array of Z eigenvalues (+1/-1) per wire and basis state."""
    s = np.arange(2**n)
    return np.array([1 - 2 * ((s >> (n - 1 - i)) & 1) for i in range(n)], dtype=np.int8)


def bonds(n: int, periodic: bool = True):
    nn = [(i, (i + 1) % n) for i in range(n if periodic else n - 1)]
    nnn = [(i, (i + 2) % n) for i in range(n if periodic else n - 2)]
    return nn, nnn


@lru_cache(maxsize=None)
def _diag_parts(n: int, periodic: bool):
    z = zsigns(n).astype(np.float64)
    nn, nnn = bonds(n, periodic)
    d_nn = sum(z[i] * z[j] for i, j in nn)
    d_nnn = sum(z[i] * z[j] for i, j in nnn)
    return d_nn, d_nnn


@lru_cache(maxsize=None)
def _xsum(n: int) -> sp.csr_matrix:
    dim = 2**n
    rows, cols = [], []
    s = np.arange(dim)
    for i in range(n):
        rows.append(s)
        cols.append(s ^ (1 << (n - 1 - i)))
    data = np.ones(n * dim)
    return sp.csr_matrix((data, (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))


def hamiltonian_sparse(n: int, kappa: float, h: float, periodic: bool = True) -> sp.csr_matrix:
    d_nn, d_nnn = _diag_parts(n, periodic)
    diag = -d_nn + kappa * d_nnn
    return (sp.diags(diag) - h * _xsum(n)).tocsr()


# ----------------------------------------------------------------------------
# eigen-solver
# ----------------------------------------------------------------------------

def parity_apply(n: int, v: np.ndarray) -> np.ndarray:
    """P = prod_i X_i maps |s> -> |~s>, i.e. reverses the state vector."""
    return v[::-1]


def low_spectrum(n, kappa, h, k=6, periodic=True, v0=None, method="eigsh", seed=0, tol=0.0):
    """Lowest-k eigenpairs, sorted. Returns (E, V, info)."""
    H = hamiltonian_sparse(n, kappa, h, periodic)
    dim = H.shape[0]
    if method == "dense" or dim <= 64:
        E, V = np.linalg.eigh(H.toarray())
        E, V = E[:k], V[:, :k]
        info = {"method": "dense", "nconv": k}
    else:
        rng = np.random.default_rng(seed)
        start = rng.standard_normal(dim)
        if v0 is not None:  # warm start mixed with random vector so excited states are not suppressed [S23]
            start = v0 / np.linalg.norm(v0) + 0.3 * start / np.linalg.norm(start)
        E, V = spla.eigsh(H, k=k, which="SA", v0=start, tol=tol, maxiter=20000)
        order = np.argsort(E)
        E, V = E[order], V[:, order]
        info = {"method": "eigsh", "nconv": k}
    HV = H @ V
    info["residual"] = float(np.max(np.linalg.norm(HV - V * E, axis=0)))
    info["orthogonality"] = float(np.max(np.abs(V.T @ V - np.eye(V.shape[1]))))
    return E, V, info


def ground_subspace(E, tol=1e-8):
    """Indices of eigenvalues exactly degenerate with E0 (tolerance scaled by |E0|)."""
    scale = max(1.0, abs(E[0]))
    return np.where(E - E[0] <= tol * scale)[0]


def subspace_fidelity(Va, Vb) -> float:
    """Mean principal squared fidelity between ground subspaces (columns orthonormal).
    Equals |<a|b>|^2 for 1-d subspaces; invariant to rotations inside degenerate spaces."""
    M = Va.T.conj() @ Vb
    return float(np.sum(np.abs(M) ** 2) / max(Va.shape[1], Vb.shape[1]))


# ----------------------------------------------------------------------------
# observables (pure states, mixtures, density matrices)
# ----------------------------------------------------------------------------

def zz_matrix_from_probs(n: int, probs: np.ndarray) -> np.ndarray:
    z = zsigns(n).astype(np.float64)
    return (z * probs) @ z.T


def x_expect_state(n: int, psi: np.ndarray) -> np.ndarray:
    s = np.arange(2**n)
    return np.array([np.real(np.vdot(psi, psi[s ^ (1 << (n - 1 - i))])) for i in range(n)])


def x_expect_rho(n: int, rho: np.ndarray) -> np.ndarray:
    s = np.arange(2**n)
    return np.array([np.real(np.sum(rho[s ^ (1 << (n - 1 - i)), s])) for i in range(n)])


def correlation_summary(Cij: np.ndarray, xs: np.ndarray) -> dict:
    """C(r), S(q) (q = 2 pi m/N, m = 0..N-1), q*, m_x from a ZZ matrix and <X_i>."""
    n = Cij.shape[0]
    Cr = np.array([np.mean([Cij[i, (i + r) % n] for i in range(n)]) for r in range(n)])
    idx = np.arange(n)
    q = 2 * np.pi * np.arange(n) / n
    phase = np.exp(1j * q[:, None, None] * (idx[None, :, None] - idx[None, None, :]))
    Sq = np.real(np.einsum("qij,ij->q", phase, Cij)) / n
    half = n // 2 + 1
    qstar_idx = int(np.argmax(Sq[:half]))
    return {
        "Cr": Cr, "Sq": Sq, "q": q, "qstar": q[qstar_idx], "qstar_idx": qstar_idx,
        "mx": float(np.mean(xs)), "S0": float(Sq[0]),
        "Spi2": float(Sq[n // 4]) if n % 4 == 0 else float(np.nan),
    }


def observables_state(n: int, psi: np.ndarray) -> dict:
    probs = np.abs(psi) ** 2
    Cij = zz_matrix_from_probs(n, probs)
    xs = x_expect_state(n, psi)
    out = correlation_summary(Cij, xs)
    out["Cij"] = Cij
    out["parity"] = float(np.real(np.vdot(psi, parity_apply(n, psi))))
    return out


def observables_subspace(n: int, V: np.ndarray) -> dict:
    """Observables of the equal mixture over a (degenerate) ground subspace."""
    d = V.shape[1]
    if d == 1:
        return observables_state(n, V[:, 0])
    probs = np.sum(np.abs(V) ** 2, axis=1) / d
    Cij = zz_matrix_from_probs(n, probs)
    xs = np.mean([x_expect_state(n, V[:, a]) for a in range(d)], axis=0)
    out = correlation_summary(Cij, xs)
    out["Cij"] = Cij
    out["parity"] = float(np.mean([np.real(np.vdot(V[:, a], parity_apply(n, V[:, a]))) for a in range(d)]))
    return out


def observables_rho(n: int, rho: np.ndarray) -> dict:
    probs = np.real(np.diag(rho))
    Cij = zz_matrix_from_probs(n, probs)
    xs = x_expect_rho(n, rho)
    out = correlation_summary(Cij, xs)
    out["Cij"] = Cij
    out["parity"] = float(np.real(np.sum(rho[::-1, :].diagonal())))  # Tr(P rho)
    return out


def half_chain_entropy(n: int, psi: np.ndarray) -> float:
    """von Neumann entanglement entropy (natural log) of sites 0..n/2-1 of a PURE state."""
    m = psi.reshape(2 ** (n // 2), 2 ** (n - n // 2))
    s = np.linalg.svd(m, compute_uv=False) ** 2
    s = s[s > 1e-14]
    return float(-np.sum(s * np.log(s)))


# ----------------------------------------------------------------------------
# analytical overlays (approximate thermodynamic references, NOT labels)
# ----------------------------------------------------------------------------

def h_ising(kappa):
    """Second-order Ising estimate, rationalized so h_I(0) = 1 [S10], defined 0<=kappa<=1/2."""
    k = np.asarray(kappa, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt((1 - 3 * k + 4 * k**2) / (1 - k))
        v = 2 * (1 - 2 * k) / (1 + root)
    return np.where((k >= 0) & (k <= 0.5), v, np.nan)


def h_pt_lower(kappa):
    """Lower antiphase-floating reference (Pokrovsky-Talapov), empirical 1.05 fit [S13, S12]."""
    k = np.asarray(kappa, dtype=float)
    return np.where(k > 0.5, 1.05 * (k - 0.5), np.nan)


def h_kt_upper(kappa):
    """Upper floating-paramagnet reference (KT/BKT), empirical 1.05 fit [S13, S12]."""
    k = np.asarray(kappa, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(k > 0.5, 1.05 * np.sqrt(np.clip((k - 0.5) * (k - 0.1), 0, None)), np.nan)


def h_peschel_emery(kappa):
    """Disorder line inside the paramagnet (not a phase boundary) [S10]."""
    k = np.asarray(kappa, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((k > 0) & (k < 0.5), 1 / (4 * k) - k, np.nan)


def software_versions() -> dict:
    import importlib.metadata as md
    import platform
    import sys
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in ["numpy", "scipy", "pennylane", "matplotlib"]:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "absent"
    return out


def observables_rho_full(n: int, rho: np.ndarray, H=None) -> dict:
    """Observables of a density matrix, including symmetry-verified (P = prod X, s = +1)
    ZZ correlators and <X_i> [S21]: <O>_+ = (<O> + <OP>)/(1 + <P>) for [O,P]=0."""
    out = observables_rho(n, rho)
    dim = 2**n
    s = np.arange(dim)
    z = zsigns(n).astype(np.float64)
    anti = np.real(rho[dim - 1 - s, s])            # <~s|rho|s>
    ZZP = (z * anti) @ z.T                         # Tr(Z_i Z_j P rho)
    allmask = dim - 1
    XP = np.array([np.real(np.sum(rho[s ^ (allmask ^ (1 << (n - 1 - i))), s])) for i in range(n)])
    Pexp = out["parity"]
    xs = x_expect_rho(n, rho)
    Cv = (out["Cij"] + ZZP) / (1 + Pexp)
    xv = (xs + XP) / (1 + Pexp)
    ver = correlation_summary(Cv, xv)
    out["Cij_ver"] = Cv; out["Sq_ver"] = ver["Sq"]; out["mx_ver"] = ver["mx"]
    out["accept"] = (1 + Pexp) / 2                 # probability of the even-parity sector
    out["purity"] = float(np.real(np.sum(np.abs(rho) ** 2)))
    out["xs"] = xs
    if H is not None:
        out["E"] = float(np.real(np.sum(H.multiply(rho.T))) if hasattr(H, "multiply") else np.real(np.trace(H @ rho)))
    return out
