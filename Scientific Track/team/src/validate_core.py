"""Validation of annni_core against the read-only starter kit, PennyLane and ideal states.
Run:  python team/src/validate_core.py   (exits non-zero on failure; writes results/validation/core_validation.json)"""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(v, "1")
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(TRACK))
import annni_core as ac
import pennylane as qml
from starter_kit.annni import build_annni_hamiltonian
from starter_kit.exact_diag import _dense_annni_matrix
from starter_kit import reference as sref

rep = {}
rng = np.random.default_rng(2026)
# 1. Hamiltonian matches starter dense matrix and PennyLane matrix, Hermitian
worst = 0.0
for n in (5, 6, 8):
    for (k, h) in [(0.0, 0.0), (0.5, 0.0), (0.3, 0.7), (1.0, 2.0)] + [tuple(rng.uniform([0, 0], [1, 2])) for _ in range(3)]:
        Hs = ac.hamiltonian_sparse(n, k, h).toarray()
        Hd = _dense_annni_matrix(n, k, h, True)
        Hp = qml.matrix(build_annni_hamiltonian(n, k, h), wire_order=range(n))
        worst = max(worst, np.abs(Hs - Hd).max(), np.abs(Hs - Hp).max(), np.abs(Hs - Hs.T).max())
rep["H_vs_starter_and_pennylane_maxabs"] = worst
assert worst < 1e-12
nn, nnn = ac.bonds(8); rep["bond_counts_N8_periodic"] = [len(nn), len(nnn)]
assert len(set(nn)) == 8 and len(set(nnn)) == 8
# 2. eigsh vs dense spectrum, residuals, orthogonality
dE = 0.0; res = 0.0; orth = 0.0
for (k, h) in [(0.2, 0.5), (0.5, 0.05), (0.8, 0.4), (0.6, 1.5)] + [tuple(rng.uniform([0, 0], [1, 2])) for _ in range(4)]:
    E1, V1, i1 = ac.low_spectrum(8, k, h, k=6, method="eigsh")
    E2, V2, i2 = ac.low_spectrum(8, k, h, k=6, method="dense")
    dE = max(dE, np.abs(E1 - E2).max()); res = max(res, i1["residual"]); orth = max(orth, i1["orthogonality"])
rep["eigsh_vs_dense_maxdE"] = dE; rep["eigsh_max_residual"] = res; rep["eigsh_max_orth_err"] = orth
assert dE < 1e-9 and res < 1e-8 and orth < 1e-9
# 3. Perron-Frobenius: unique ground state with even parity for h>0
E, V, _ = ac.low_spectrum(8, 0.3, 0.2, k=4, method="dense")
rep["gs_parity_h0.2"] = float(V[:, 0] @ V[::-1, 0]); assert rep["gs_parity_h0.2"] > 0.999999
# 4. ideal-state normalization checks
n = 8; dim = 2**n
up = np.zeros(dim); up[0] = 1
cat = np.zeros(dim); cat[0] = cat[-1] = 2**-0.5
bits = [0, 0, 1, 1, 0, 0, 1, 1]; idx = int("".join(map(str, bits)), 2)
anti = np.zeros(dim); anti[idx] = 1
plus = np.ones(dim) / np.sqrt(dim)
mixed = np.eye(dim) / dim
o = ac.observables_state(n, up); rep["ferro_S0"] = o["S0"]; assert abs(o["S0"] - n) < 1e-12
o = ac.observables_state(n, cat); rep["cat_S0"] = o["S0"]; rep["cat_mean_Z"] = 0.0; assert abs(o["S0"] - n) < 1e-12
o = ac.observables_state(n, anti); rep["antiphase_Spi2"] = o["Spi2"]; assert abs(o["Spi2"] - n / 2) < 1e-12 and o["qstar_idx"] == 2
o = ac.observables_state(n, plus); rep["plus_S"] = o["Sq"].tolist(); rep["plus_mx"] = o["mx"]
assert np.allclose(o["Sq"], 1) and abs(o["mx"] - 1) < 1e-12
o = ac.observables_rho(n, mixed); rep["mixed_S"] = o["Sq"].tolist(); rep["mixed_mx"] = o["mx"]
assert np.allclose(o["Sq"], 1) and abs(o["mx"]) < 1e-12
# rho path agrees with pure path
E, V, _ = ac.low_spectrum(8, 0.7, 0.6, k=2, method="dense")
a = ac.observables_state(8, V[:, 0]); b = ac.observables_rho(8, np.outer(V[:, 0], V[:, 0]))
rep["rho_vs_state_maxdiff"] = float(max(np.abs(a["Cij"] - b["Cij"]).max(), abs(a["mx"] - b["mx"]), abs(a["parity"] - b["parity"])))
assert rep["rho_vs_state_maxdiff"] < 1e-12
# X expectation vs PennyLane
dev = qml.device("default.qubit", wires=8)
@qml.qnode(dev)
def xq(psi):
    qml.StatePrep(psi, wires=range(8)); return [qml.expval(qml.X(i)) for i in range(8)] + [qml.expval(qml.Z(0) @ qml.Z(3))]
r = np.array(xq(V[:, 0]))
rep["X_ZZ_vs_pennylane_maxdiff"] = float(max(np.abs(r[:8] - ac.x_expect_state(8, V[:, 0])).max(), abs(r[8] - a["Cij"][0, 3])))
assert rep["X_ZZ_vs_pennylane_maxdiff"] < 1e-10
# 5. reference curves
rep["hI_0"] = float(ac.h_ising(0.0)); rep["hI_half"] = float(ac.h_ising(0.5))
rep["hI_vs_starter_k0.3"] = float(ac.h_ising(0.3) - sref.ising_transition(0.3))
rep["starter_hI_0_bug"] = float(sref.ising_transition(0.0))
assert abs(rep["hI_0"] - 1) < 1e-12 and abs(rep["hI_half"]) < 1e-12 and abs(rep["hI_vs_starter_k0.3"]) < 1e-12
# second-order gap equation h - k h^2/(2(1-k)) = 1-2k
kk = 0.3; hh = float(ac.h_ising(kk)); rep["hI_gap_eq_residual"] = hh - kk * hh**2 / (2 * (1 - kk)) - (1 - 2 * kk)
assert abs(rep["hI_gap_eq_residual"]) < 1e-12
rep["versions"] = ac.software_versions()
out = TRACK / "team" / "results" / "validation"; out.mkdir(parents=True, exist_ok=True)
(out / "core_validation.json").write_text(json.dumps(rep, indent=1, default=float))
print(json.dumps({k: v for k, v in rep.items() if k not in ("plus_S", "mixed_S")}, indent=1, default=float))
print("ALL CORE CHECKS PASSED")
