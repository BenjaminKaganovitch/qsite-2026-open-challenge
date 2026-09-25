"""N-qubit RY/CNOT-ring ansatz: fast NumPy engines (statevector + adjoint gradient,
density matrix with the challenge noise model) and the equivalent PennyLane QNodes.

Ansatz 'hea_ry_ring' with L entangling layers, params shape (L+1, N):
    RY(theta[0, i]) on every wire
    repeat l = 1..L:
        CNOT(i, (i+1) % N) for i = 0..N-1       (sequential ring, N CNOTs per layer)
        [noise]  DepolarizingChannel(p) on the CNOT *target* immediately after every CNOT
        RY(theta[l, i]) on every wire
Total CNOTs = channels = L*N. The NumPy engines are validated against PennyLane
default.qubit / default.mixed in validate_circuits.py; PennyLane is used for the
reported noisy evaluations."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def ops_list(n: int, L: int):
    ops = [("RY", i, (0, i)) for i in range(n)]
    for l in range(1, L + 1):
        for i in range(n):
            ops.append(("CNOT", (i, (i + 1) % n), None))
        ops += [("RY", i, (l, i)) for i in range(n)]
    return ops


@lru_cache(maxsize=None)
def cnot_perm(n: int, c: int, t: int) -> np.ndarray:
    s = np.arange(2**n)
    cb = (s >> (n - 1 - c)) & 1
    return s ^ (cb << (n - 1 - t))


@lru_cache(maxsize=None)
def flip_perm(n: int, t: int) -> np.ndarray:
    return np.arange(2**n) ^ (1 << (n - 1 - t))


@lru_cache(maxsize=None)
def zsign(n: int, t: int) -> np.ndarray:
    s = np.arange(2**n)
    return (1 - 2 * ((s >> (n - 1 - t)) & 1)).astype(np.float64)


def _ry_vec(psi, n, w, theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    v = psi.reshape(2**w, 2, -1)
    a, b = v[:, 0, :], v[:, 1, :]
    out = np.empty_like(v)
    out[:, 0, :] = c * a - s * b
    out[:, 1, :] = s * a + c * b
    return out.reshape(-1)


def _dry_vec(psi, n, w, theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    v = psi.reshape(2**w, 2, -1)
    a, b = v[:, 0, :], v[:, 1, :]
    out = np.empty_like(v)
    out[:, 0, :] = 0.5 * (-s * a - c * b)
    out[:, 1, :] = 0.5 * (c * a - s * b)
    return out.reshape(-1)


def statevector(theta: np.ndarray, n: int, L: int) -> np.ndarray:
    psi = np.zeros(2**n); psi[0] = 1.0
    for kind, w, pi in ops_list(n, L):
        if kind == "RY":
            psi = _ry_vec(psi, n, w, theta[pi])
        else:
            psi = psi[cnot_perm(n, *w)]
    return psi


def energy_and_grad(theta_flat, H, n, L):
    theta = theta_flat.reshape(L + 1, n)
    ops = ops_list(n, L)
    psi = statevector(theta, n, L)
    lam = H @ psi
    E = float(psi @ lam)
    grad = np.zeros_like(theta)
    for kind, w, pi in reversed(ops):
        if kind == "RY":
            psi = _ry_vec(psi, n, w, -theta[pi])
            grad[pi] = 2.0 * float(lam @ _dry_vec(psi, n, w, theta[pi]))
            lam = _ry_vec(lam, n, w, -theta[pi])
        else:
            perm = cnot_perm(n, *w)
            psi = psi[perm]; lam = lam[perm]
    return E, grad.reshape(-1)


# ---------------- density-matrix engine with target depolarization ---------------

def _ry_rho(rho, n, w, theta):
    rho = _ry_vec_cols(rho, n, w, theta)
    return _ry_vec_cols(rho.T, n, w, theta).T


def _ry_vec_cols(M, n, w, theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    v = M.reshape(2**w, 2, -1)
    a, b = v[:, 0, :], v[:, 1, :]
    out = np.empty_like(v)
    out[:, 0, :] = c * a - s * b
    out[:, 1, :] = s * a + c * b
    return out.reshape(M.shape)


def depolarize(rho, n, t, p):
    """PennyLane DepolarizingChannel(p) on wire t: (1-p)rho + p/3 (XrhoX + YrhoY + ZrhoZ)."""
    if p == 0:
        return rho
    f = flip_perm(n, t); z = zsign(n, t)
    X = rho[f][:, f]
    Zr = rho * np.outer(z, z)
    Y = Zr[f][:, f]
    return (1 - p) * rho + (p / 3) * (X + Y + Zr)


def density_matrix(theta, n, L, p, p_scale_locations=None):
    """Noisy circuit; channel after each CNOT target with strength p (or per-location list)."""
    rho = np.zeros((2**n, 2**n)); rho[0, 0] = 1.0
    g = 0
    for kind, w, pi in ops_list(n, L):
        if kind == "RY":
            rho = _ry_rho(rho, n, w, theta[pi])
        else:
            perm = cnot_perm(n, *w)
            rho = rho[perm][:, perm]
            pg = p if p_scale_locations is None else p_scale_locations[g]
            rho = depolarize(rho, n, w[1], pg)
            g += 1
    return rho


# ---------------- PennyLane equivalents ----------------

def pennylane_qnode(n, L, p, device="default.mixed"):
    import pennylane as qml
    dev = qml.device(device, wires=n)

    @qml.qnode(dev, interface="numpy")
    def circ(theta):
        for i in range(n):
            qml.RY(theta[0, i], wires=i)
        for l in range(1, L + 1):
            for i in range(n):
                t = (i + 1) % n
                qml.CNOT(wires=[i, t])
                if device == "default.mixed":
                    qml.DepolarizingChannel(p, wires=t)
            for i in range(n):
                qml.RY(theta[l, i], wires=i)
        if device == "default.mixed":
            return qml.density_matrix(wires=range(n))
        return qml.state()
    return circ


def plus_state_params(n, L):
    th = np.zeros((L + 1, n)); th[0, :] = np.pi / 2
    return th


def product_params(n, L, bits):
    th = np.zeros((L + 1, n)); th[0, :] = np.pi * np.asarray(bits)
    return th
