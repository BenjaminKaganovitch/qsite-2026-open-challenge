"""Clean (p=0) VQE for the RY/CNOT-ring ansatz with exact adjoint gradients (L-BFGS-B)."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
import annni_core as ac
import circuits as cc


def optimize(H, n, L, theta0, maxiter=3000, gtol=1e-9):
    r = minimize(cc.energy_and_grad, theta0.reshape(-1), args=(H, n, L), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-14})
    return r.x.reshape(L + 1, n), float(r.fun), int(r.nit)


def initial_candidates(n, L, rng, n_random=1):
    c = {"zero": np.zeros((L + 1, n)),                                  # |00..0> (ferro)
         "plus": cc.plus_state_params(n, L),                            # |++..+> (paramagnet)
         "anti": cc.product_params(n, L, [(i // 2) % 2 for i in range(n)])}  # |0011..> (antiphase)
    for r in range(n_random):
        c[f"rand{r}"] = rng.uniform(-np.pi, np.pi, (L + 1, n))
    for k in c:  # tiny symmetric-breaking jitter so gradients are nonzero at product states
        c[k] = c[k] + 1e-3 * rng.standard_normal(c[k].shape)
    return c


def manifold_fidelity(psi, E, V, window=1e-3):
    """|<gs|psi>|^2 and the fidelity with the low-energy manifold {E_a - E0 < window}
    (captures a symmetry-broken state inside an ordered-phase tunnelling doublet)."""
    ov = np.abs(V.T.conj() @ psi) ** 2
    sel = (E - E[0]) < window
    return float(ov[0]), float(np.sum(ov[sel])), int(np.sum(sel))
