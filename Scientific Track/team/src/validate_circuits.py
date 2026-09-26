"""Validate NumPy circuit engines against PennyLane (default.qubit / default.mixed), adjoint
gradients against finite differences, and channel placement/count in the PennyLane tape."""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(v, "1")
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import pennylane as qml
import annni_core as ac
import circuits as cc

rep = {}; rng = np.random.default_rng(7)
n = 8
for L in (2, 4):
    th = rng.uniform(-np.pi, np.pi, (L + 1, n))
    psi = cc.statevector(th, n, L)
    psi_pl = np.array(cc.pennylane_qnode(n, L, 0.0, "default.qubit")(th))
    rep[f"L{L}_state_vs_default.qubit"] = float(np.abs(psi - psi_pl).max())
    H = ac.hamiltonian_sparse(n, 0.4, 0.8)
    E, g = cc.energy_and_grad(th.reshape(-1), H, n, L)
    eps = 1e-6; fd = []
    for k in range(0, th.size, 5):
        e = np.zeros(th.size); e[k] = eps
        fd.append((cc.energy_and_grad(th.reshape(-1) + e, H, n, L)[0] - cc.energy_and_grad(th.reshape(-1) - e, H, n, L)[0]) / (2 * eps))
    rep[f"L{L}_adjoint_vs_fd"] = float(np.abs(np.array(fd) - g[::5]).max())
    for p in (0.0, 0.01, 0.05):
        t0 = time.time(); rho_pl = np.array(cc.pennylane_qnode(n, L, p)(th)); tpl = time.time() - t0
        t0 = time.time(); rho = cc.density_matrix(th, n, L, p); tnp = time.time() - t0
        rep[f"L{L}_p{p}_rho_vs_default.mixed"] = float(np.abs(rho - rho_pl).max())
        rep[f"L{L}_p{p}_time_pl_s"] = tpl; rep[f"L{L}_p{p}_time_numpy_s"] = tnp
    rep[f"L{L}_p0_mixed_vs_pure"] = float(np.abs(np.array(cc.pennylane_qnode(n, L, 0.0)(th)) - np.outer(psi, psi)).max())
    # tape inspection: every CNOT followed immediately by DepolarizingChannel on its target
    tape = qml.workflow.construct_tape(cc.pennylane_qnode(n, L, 0.05))(th)
    ops = tape.operations
    ncnot = sum(o.name == "CNOT" for o in ops); nch = sum(o.name == "DepolarizingChannel" for o in ops)
    ok = all(ops[i + 1].name == "DepolarizingChannel" and ops[i + 1].wires[0] == ops[i].wires[1]
             and abs(float(ops[i + 1].parameters[0]) - 0.05) < 1e-15 for i, o in enumerate(ops) if o.name == "CNOT")
    only_after = all(ops[i - 1].name == "CNOT" for i, o in enumerate(ops) if o.name == "DepolarizingChannel")
    rep[f"L{L}_cnots"] = ncnot; rep[f"L{L}_channels"] = nch; rep[f"L{L}_placement_ok"] = bool(ok and only_after)
    assert ncnot == n * L and nch == n * L and ok and only_after
    assert rep[f"L{L}_state_vs_default.qubit"] < 1e-10 and rep[f"L{L}_adjoint_vs_fd"] < 1e-6
    assert max(rep[f"L{L}_p{p}_rho_vs_default.mixed"] for p in (0.0, 0.01, 0.05)) < 1e-10
# HVA ansatz actually used for the reported diagrams: tape check at N=8, L=4, p=0.05
import hva
g_hva = rng.uniform(-1, 1, (4, 3))
tape = qml.workflow.construct_tape(hva.pennylane_qnode(8, 4, 0.05))(g_hva)
ops = tape.operations
n_cnot = sum(o.name == "CNOT" for o in ops)
n_ch = sum(o.name == "DepolarizingChannel" for o in ops)
chan_ok = all(i > 0 and ops[i - 1].name == "CNOT" and o.wires[0] == ops[i - 1].wires[1]
              and abs(float(o.parameters[0]) - 0.05) < 1e-15
              for i, o in enumerate(ops) if o.name == "DepolarizingChannel")
every_cnot = all(i + 1 < len(ops) and ops[i + 1].name == "DepolarizingChannel" for i, o in enumerate(ops) if o.name == "CNOT")
other_noise = [o.name for o in ops if o.name not in ("CNOT", "DepolarizingChannel", "Hadamard", "RZ", "RX")]
ps = sorted({float(o.parameters[0]) for o in ops if o.name == "DepolarizingChannel"})
rep["HVA_L4_cnots"] = n_cnot; rep["HVA_L4_channels"] = n_ch; rep["HVA_L4_p"] = ps
rep["HVA_L4_placement_ok"] = bool(chan_ok and every_cnot and not other_noise)
rep["HVA_L4_other_ops"] = other_noise
assert n_cnot == 128 and n_ch == 128 and chan_ok and every_cnot and not other_noise and ps == [0.05]
# terminal-channel benchmark [S28]: channel after final state on every site -> C_ij scale a^2
th = rng.uniform(-np.pi, np.pi, (3, n)); rho = cc.density_matrix(th, n, 2, 0.0)
o0 = ac.observables_rho(n, rho)
for p in (0.01, 0.05):
    r = rho.copy()
    for t in range(n):
        r = cc.depolarize(r, n, t, p)
    o = ac.observables_rho(n, r); a = 1 - 4 * p / 3
    off = ~np.eye(n, dtype=bool)
    rep[f"terminal_p{p}_Cij_ratio_err"] = float(np.abs(o["Cij"][off] - a**2 * o0["Cij"][off]).max())
    rep[f"terminal_p{p}_Sq_formula_err"] = float(np.abs(o["Sq"] - (1 + a**2 * (o0["Sq"] - 1))).max())
    rep[f"terminal_p{p}_mx_err"] = float(abs(o["mx"] - a * o0["mx"]))
    assert rep[f"terminal_p{p}_Sq_formula_err"] < 1e-12
out = HERE.parent / "results" / "validation"; out.mkdir(parents=True, exist_ok=True)
(out / "circuit_validation.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1)); print("ALL CIRCUIT CHECKS PASSED")
