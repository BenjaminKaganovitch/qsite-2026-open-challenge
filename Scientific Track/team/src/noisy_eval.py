"""Fixed-parameter noisy evaluation of the clean-optimised HVA circuits (CLI).

--mode pennylane : PennyLane default.mixed at p in {0.01, 0.05} with DepolarizingChannel(p) on the
                   target after every CNOT; p=0 evaluated as the exact pure circuit state.
                   -> team/results/noisy_hva_N8_L4_31x31/
--mode extras    : validated NumPy density-matrix engine: linear ZNE points (lambda*p, lambda=2,3),
                   composed-channel points p_lambda=(3/4)[1-(1-4p/3)^lambda] (p=0.05), and the exact
                   first-order response d<O>/dp|_0 by central difference (+-dp) [S28].
                   -> team/results/noisy_extras_hva_N8_L4_31x31/
Parameters, circuit, depth and grid are identical for every p (fixed-parameter protocol)."""
from scan_io import set_single_thread
set_single_thread()
import argparse, os, socket, time
import numpy as np
import annni_core as ac
import hva
import scan_io as sio

KEYS = ["Cij", "Sq", "Cr", "mx", "qstar", "S0", "Spi2", "parity", "Cij_ver", "Sq_ver", "mx_ver", "accept", "purity", "E"]


_a = lambda p: 1 - 4 * p / 3
PTS = {"lin_0.02": 0.02, "lin_0.03": 0.03, "lin_0.1": 0.10, "lin_0.15": 0.15,
       "comp_0.05_l2": 0.75 * (1 - _a(0.05) ** 2), "comp_0.05_l3": 0.75 * (1 - _a(0.05) ** 3),
       "plus_dp": 1e-4, "minus_dp": -1e-4}


def pack(res_list):
    return {k: np.asarray([r[k] for r in res_list]) for k in KEYS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pennylane", "extras", "numpy"], required=True)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--nk", type=int, default=31); ap.add_argument("--nh", type=int, default=31)
    ap.add_argument("--rows", default="all"); ap.add_argument("--source", default=None)
    ap.add_argument("--dp", type=float, default=1e-4)
    args = ap.parse_args()
    src = args.source or f"vqe_hva_N{args.n}_L{args.L}_{args.nk}x{args.nh}"
    tag = {"pennylane": "noisy", "extras": "noisy_extras", "numpy": "noisy_np"}[args.mode]
    scan = f"{tag}_hva_N{args.n}_L{args.L}_{args.nk}x{args.nh}" if args.source is None else f"{tag}__{src}"
    n, L = args.n, args.L
    core = {"scan": scan, "source_scan": src, "N": n, "L": L, "nk": args.nk, "nh": args.nh, "mode": args.mode,
            "noise_model": "qml.DepolarizingChannel(p) on CNOT target immediately after every CNOT; 4NL channels",
            "protocol": "fixed parameters from clean (p=0) optimisation; identical circuit/depth/grid for all p",
            "device": "default.mixed (PennyLane 0.44.1)" if args.mode == "pennylane" else "NumPy DM engine validated vs default.mixed to 1e-13",
            "parity_verification": "s=+1, P=prod X, <O>_+=(<O>+<OP>)/(1+<P>)",
            "zne_linear_lambdas": [1, 2, 3], "composed_lambdas_p0.05": [1, 2, 3], "dp": args.dp}
    rows = sio.parse_rows(args.rows, args.nk)
    prog = sio.Progress(len(rows), scan)
    if args.mode == "pennylane":
        qn = {p: hva.pennylane_qnode(n, L, p) for p in (0.01, 0.05)}
    for r in rows:
        if sio.row_done(scan, r, core):
            prog.step(f"row {r} exists"); continue
        t0 = time.time()
        with np.load(sio.row_path(src, r)) as z:
            params, hs, kap = z["params"], z["h"], float(z["kappa"])
        arrays = {"h": hs, "kappa": np.array(kap)}
        Hs = [ac.hamiltonian_sparse(n, kap, h) for h in hs]
        ptag = sio.RESULTS / scan / f".partial_row_{r:03d}.npz"
        done_pts = []
        if ptag.exists():
            try:
                with np.load(ptag, allow_pickle=True) as zz:
                    done_pts = list(zz["pts"])
            except Exception:
                done_pts = []
        for j in range(len(done_pts), len(hs)):
            rec = {}
            if args.mode in ("pennylane", "numpy"):
                for p in (0.0, 0.01, 0.05):
                    if p == 0.0:
                        psi = hva.state(params[j], n, L); rho = np.outer(psi, psi.conj())
                    elif args.mode == "numpy":
                        rho = hva.gate_density_matrix(params[j], n, L, p)
                    else:
                        rho = np.array(qn[p](params[j]))
                    o = ac.observables_rho_full(n, rho, Hs[j])
                    for k in KEYS:
                        rec[f"p{p}_{k}"] = o[k]
            else:
                for name, p in PTS.items():
                    o = ac.observables_rho_full(n, hva.gate_density_matrix(params[j], n, L, p), Hs[j])
                    for k in KEYS:
                        rec[f"{name}_{k}"] = o[k]
            done_pts.append(rec)
            (sio.RESULTS / scan).mkdir(parents=True, exist_ok=True)
            tmp = sio.RESULTS / scan / f".tmp_partial_{r:03d}_{os.getpid()}.npz"
            with open(tmp, "wb") as f:
                np.savez(f, pts=np.array(done_pts, dtype=object))
            os.replace(tmp, ptag)
        for k in done_pts[0]:
            arrays[k] = np.asarray([d[k] for d in done_pts])
        if args.mode == "extras":
            for name, p in PTS.items():
                arrays[f"{name}_pvalue"] = np.array(p)
        meta = dict(core, row=r, kappa=kap, runtime_s=time.time() - t0, host=socket.gethostname(),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), versions=ac.software_versions(), peak_rss_mb=sio.peak_rss_mb())
        sio.save_row(scan, r, meta, arrays)
        ptag.unlink(missing_ok=True)
        prog.step(f"row {r} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
