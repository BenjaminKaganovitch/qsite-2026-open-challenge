"""Hamiltonian-variational ansatz (HVA) for the periodic ANNNI ring.

    |psi(g)> = prod_{l=1..L} [ exp(-i b_l sum_i X_i) exp(-i c_l sum_i Z_iZ_{i+2}) exp(-i a_l sum_i Z_iZ_{i+1}) ] |+>^N

Parameters g = (a_l, c_l, b_l)_{l=1..L}: 3L numbers, translation- and parity-symmetric.
Gate compilation (used for PennyLane and the noise model):
    Hadamard on every wire                                   (prepares |+>^N, no CNOT)
    per layer: for each NN bond (i,i+1), then each NNN bond (i,i+2), i = 0..N-1:
        CNOT(i,j) ; RZ(2*angle) on j ; CNOT(i,j)              == exp(-i angle Z_i Z_j)
    then RX(2 b_l) on every wire                              == exp(-i b_l X)
CNOTs per layer = 2 * (2N) = 4N; each is followed by DepolarizingChannel(p) on its target.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

import annni_core as ac
import circuits as cc


# ---------------- fast exact simulation (diagonal phases + X-mixer) -------------

def _rx_all(psi, n, beta):
    c, s = np.cos(beta), -1j * np.sin(beta)
    for w in range(n):
        v = psi.reshape(2**w, 2, -1)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = c * a + s * b
        v[:, 1, :] = s * a + c * b
    return psi


@lru_cache(maxsize=None)
def _diag(n):
    return ac._diag_parts(n, True)


def _xsum_apply(psi, n):
    out = np.zeros_like(psi)
    s = np.arange(2**n)
    for w in range(n):
        out += psi[s ^ (1 << (n - 1 - w))]
    return out


def state(g, n, L):
    dnn, dnnn = _diag(n)
    psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    g = np.asarray(g).reshape(L, 3)
    for a, c, b in g:
        psi = psi * np.exp(-1j * (a * dnn + c * dnnn))
        psi = _rx_all(psi, n, b)
    return psi


def energy_and_grad(gflat, H, n, L):
    dnn, dnnn = _diag(n)
    g = gflat.reshape(L, 3)
    psi = state(g, n, L)
    lam = H @ psi
    E = float(np.real(np.vdot(psi, lam)))
    grad = np.zeros((L, 3))
    for l in range(L - 1, -1, -1):
        a, c, b = g[l]
        # undo X-mixer; generator Xsum
        grad[l, 2] = 2 * np.real(np.vdot(lam, -1j * _xsum_apply(psi, n)))
        psi = _rx_all(psi, n, -b); lam = _rx_all(lam, n, -b)
        grad[l, 1] = 2 * np.real(np.vdot(lam, -1j * dnnn * psi))
        grad[l, 0] = 2 * np.real(np.vdot(lam, -1j * dnn * psi))
        ph = np.exp(1j * (a * dnn + c * dnnn))
        psi = psi * ph; lam = lam * ph
    return E, grad.reshape(-1)


def optimize(H, n, L, g0, maxiter=4000):
    r = minimize(energy_and_grad, np.asarray(g0).reshape(-1), args=(H, n, L), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "gtol": 1e-10, "ftol": 1e-15})
    return r.x.reshape(L, 3), float(r.fun), int(r.nit)


# ---------------- gate-level compilation ----------------

def gate_list(n, L):
    """[(name, wires, param_index or None, multiplier)]; param index into flattened (L,3)."""
    nn, nnn = ac.bonds(n, True)
    ops = [("H", (w,), None, 0.0) for w in range(n)]
    for l in range(L):
        for (bl, col) in ((nn, 0), (nnn, 1)):
            for (i, j) in bl:
                ops.append(("CNOT", (i, j), None, 0.0))
                ops.append(("RZ", (j,), 3 * l + col, 2.0))
                ops.append(("CNOT", (i, j), None, 0.0))
        ops += [("RX", (w,), 3 * l + 2, 2.0) for w in range(n)]
    return ops


def n_cnots(n, L):
    return 4 * n * L


_HAD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _mat(name, ang):
    if name == "H":
        return _HAD
    if name == "RZ":
        return np.array([[np.exp(-0.5j * ang), 0], [0, np.exp(0.5j * ang)]])
    if name == "RX":
        c, s = np.cos(ang / 2), -1j * np.sin(ang / 2)
        return np.array([[c, s], [s, c]])
    raise ValueError(name)


def _apply1(M, n, w, U):
    v = M.reshape(2**w, 2, -1)
    out = np.einsum("ab,xby->xay", U, v)
    return out.reshape(M.shape)


def gate_state(g, n, L):
    g = np.asarray(g).reshape(-1)
    psi = np.zeros(2**n, dtype=complex); psi[0] = 1
    for name, wires, pi, mult in gate_list(n, L):
        if name == "CNOT":
            psi = psi[cc.cnot_perm(n, *wires)]
        else:
            psi = _apply1(psi, n, wires[0], _mat(name, 0.0 if pi is None else mult * g[pi]))
    return psi


def gate_density_matrix(g, n, L, p, p_list=None):
    """Noisy density matrix: DepolarizingChannel(p) on the target after every CNOT.
    p_list (length n_cnots) overrides per-location strengths (used for ZNE/insertion response)."""
    g = np.asarray(g).reshape(-1)
    rho = np.zeros((2**n, 2**n), dtype=complex); rho[0, 0] = 1
    k = 0
    for name, wires, pi, mult in gate_list(n, L):
        if name == "CNOT":
            perm = cc.cnot_perm(n, *wires)
            rho = rho[perm][:, perm]
            pk = p if p_list is None else p_list[k]
            rho = cc.depolarize(rho, n, wires[1], pk)
            k += 1
        else:
            U = _mat(name, 0.0 if pi is None else mult * g[pi])
            rho = _apply1(rho, n, wires[0], U)
            rho = _apply1(rho.T, n, wires[0], U.conj()).T
    return rho


def pennylane_qnode(n, L, p, device="default.mixed"):
    import pennylane as qml
    dev = qml.device(device, wires=n)
    ops = gate_list(n, L)

    @qml.qnode(dev, interface="numpy")
    def circ(g):
        g = np.asarray(g).reshape(-1)
        for name, wires, pi, mult in ops:
            if name == "H":
                qml.Hadamard(wires=wires[0])
            elif name == "CNOT":
                qml.CNOT(wires=list(wires))
                if device == "default.mixed":
                    qml.DepolarizingChannel(p, wires=wires[1])
            elif name == "RZ":
                qml.RZ(mult * g[pi], wires=wires[0])
            elif name == "RX":
                qml.RX(mult * g[pi], wires=wires[0])
        return qml.density_matrix(wires=range(n)) if device == "default.mixed" else qml.state()
    return circ


def noisy_energy_and_grad(gflat, Hdense, n, L, p):
    """Tr(H rho_p(g)) and its exact gradient for the noisy HVA circuit (Heisenberg-picture adjoint:
    the depolarizing channel is unital and self-adjoint, so H is pulled back through channels by the
    same map). Used for the noise-aware re-optimisation protocol."""
    g = np.asarray(gflat).reshape(-1)
    ops = gate_list(n, L)
    rho = np.zeros((2**n, 2**n), dtype=complex); rho[0, 0] = 1
    saved = []
    for name, wires, pi, mult in ops:
        if name == "CNOT":
            perm = cc.cnot_perm(n, *wires)
            rho = cc.depolarize(rho[perm][:, perm], n, wires[1], p)
        else:
            if pi is not None:
                saved.append(rho)
            U = _mat(name, 0.0 if pi is None else mult * g[pi])
            rho = _apply1(rho, n, wires[0], U)
            rho = _apply1(rho.T, n, wires[0], U.conj()).T
    E = float(np.real(np.sum(Hdense.T * rho)))
    O = Hdense.astype(complex)
    grad = np.zeros_like(g, dtype=float)
    for name, wires, pi, mult in reversed(ops):
        if name == "CNOT":
            perm = cc.cnot_perm(n, *wires)
            O = cc.depolarize(O, n, wires[1], p)       # adjoint of the channel
            O = O[perm][:, perm]                        # CNOT is a self-inverse permutation
        else:
            ang = 0.0 if pi is None else mult * g[pi]
            U = _mat(name, ang)
            if pi is not None:
                rb = saved.pop()
                # dU/dtheta = mult * (-i/2) G U with G = Z (RZ) or X (RX)
                G = np.array([[1, 0], [0, -1]], dtype=complex) if name == "RZ" else np.array([[0, 1], [1, 0]], dtype=complex)
                dU = mult * (-0.5j) * (G @ U)
                A = _apply1(rb, n, wires[0], dU)
                A = _apply1(A.T, n, wires[0], U.conj()).T     # dU rho U^dagger
                grad[pi] += 2 * np.real(np.sum(O.T * A))
            # O <- U^dagger O U
            Ud = U.conj().T
            O = _apply1(O, n, wires[0], Ud)
            O = _apply1(O.T, n, wires[0], U).T
    return E, grad


def optimize_noisy(Hdense, n, L, p, g0, maxiter=200):
    r = minimize(noisy_energy_and_grad, np.asarray(g0).reshape(-1), args=(Hdense, n, L, p), jac=True,
                 method="L-BFGS-B", options={"maxiter": maxiter, "gtol": 1e-8})
    return r.x.reshape(L, 3), float(r.fun), int(r.nit)
