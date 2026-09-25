"""Noise-aware re-optimisation (Phase B protocol, explicitly distinct from fixed-parameter degradation):
at each p, minimise Tr(H rho_p(g)) over the 12 HVA parameters with the exact noisy gradient, starting
from (i) the clean optimum at the same point and (ii) the previous h's noisy optimum; keep the lower.

Example:  python team/src/reopt_scan.py --rows 6,24
Writes    team/results/reopt_hva_N8_L4_31x31_s2/row_<iii>.npz (h = every 2nd grid point)"""
from scan_io import set_single_thread
set_single_thread()
import argparse, os, socket, time
import numpy as np
import annni_core as ac
import hva
import scan_io as sio

KEYS = ["Cij", "Sq", "mx", "parity", "purity", "E"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True); ap.add_argument("--ps", default="0.01,0.05")
    ap.add_argument("--hstride", type=int, default=2); ap.add_argument("--maxiter", type=int, default=150)
    args = ap.parse_args()
    n, L = 8, 4
    ps = [float(x) for x in args.ps.split(",")]
    src = "vqe_hva_N8_L4_31x31"
    scan = f"reopt_hva_N8_L4_31x31_s{args.hstride}"
    core = {"scan": scan, "source_scan": src, "N": n, "L": L, "ps": ps, "hstride": args.hstride, "maxiter": args.maxiter,
            "protocol": "noise-aware re-optimisation of Tr(H rho_p) (exact adjoint gradient through channels), per p",
            "noise_model": "DepolarizingChannel(p) on CNOT target after every CNOT (NumPy DM engine = default.mixed to 3e-14)"}
    rows = sio.parse_rows(args.rows, 31)
    for r in rows:
        if sio.row_done(scan, r, core):
            print("row", r, "exists"); continue
        t0 = time.time()
        with np.load(sio.row_path(src, r)) as z:
            P0, hs_all, kap = z["params"], z["h"], float(z["kappa"])
        idx = list(range(0, len(hs_all), args.hstride))
        part = sio.RESULTS / scan / f".partial_row_{r:03d}.npz"
        recs = []
        if part.exists():
            with np.load(part, allow_pickle=True) as zz:
                recs = list(zz["pts"])
        todo = [(p, j) for p in ps for j in idx]
        for (p, j) in todo[len(recs):]:
            Hs = ac.hamiltonian_sparse(n, kap, hs_all[j]); Hd = Hs.toarray()
            cands = [P0[j]]
            prev = [rc for rc in recs if rc["p"] == p]
            if prev:
                cands.append(prev[-1]["params"])
            best = None
            for g0 in cands:
                g, e, it = hva.optimize_noisy(Hd, n, L, p, g0, maxiter=args.maxiter)
                if best is None or e < best[1]:
                    best = (g, e, it)
            g = best[0]
            rho = hva.gate_density_matrix(g, n, L, p)
            o = ac.observables_rho_full(n, rho, Hs)
            rec = {"p": p, "h_index": j, "h": float(hs_all[j]), "params": g, "E_noisy_reopt": best[1],
                   "E_noisy_fixed": float(np.real(np.sum(Hd.T * hva.gate_density_matrix(P0[j], n, L, p)))),
                   "E_clean_of_reopt_params": float(np.real(np.vdot(hva.state(g, n, L), Hs @ hva.state(g, n, L))))}
            for k in KEYS:
                rec[k] = o[k]
            recs.append(rec)
            (sio.RESULTS / scan).mkdir(parents=True, exist_ok=True)
            tmp = sio.RESULTS / scan / f".tmp_partial_{r:03d}_{os.getpid()}.npz"
            with open(tmp, "wb") as f:
                np.savez(f, pts=np.array(recs, dtype=object))
            os.replace(tmp, part)
            print(f"row {r} p={p} h={hs_all[j]:.3f} E_fixed={rec['E_noisy_fixed']:.4f} E_reopt={best[1]:.4f}", flush=True)
        arrays = {"kappa": np.array(kap), "h": np.array([hs_all[j] for j in idx]), "ps": np.array(ps)}
        for p in ps:
            rr = [rc for rc in recs if rc["p"] == p]
            for k in ["params", "E_noisy_reopt", "E_noisy_fixed", "E_clean_of_reopt_params"] + KEYS:
                arrays[f"p{p}_{k}"] = np.asarray([rc[k] for rc in rr])
        meta = dict(core, row=r, kappa=kap, runtime_s=time.time() - t0, host=socket.gethostname(),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), versions=ac.software_versions(), peak_rss_mb=sio.peak_rss_mb())
        sio.save_row(scan, r, meta, arrays)
        part.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
