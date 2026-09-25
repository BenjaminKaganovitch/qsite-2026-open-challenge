"""Clean (p=0) HVA-VQE scan on the (kappa, h) grid (CLI). One kappa row = forward and reverse
warm-started h sweeps; the lower-energy branch is kept per point and branch disagreement recorded.

Example:  python team/src/vqe_scan.py --n 8 --L 4 --rows 0:31
Writes    team/results/vqe_hva_N8_L4_31x31/row_<iii>.npz"""
from scan_io import set_single_thread
set_single_thread()
import argparse, socket, time
import numpy as np
import annni_core as ac
import hva
import scan_io as sio
import vqe


def run_row(args, row, kap, hs):
    n, L = args.n, args.L
    rng = np.random.default_rng(args.seed + 1000 * row)
    nh = len(hs)
    Hs = [ac.hamiltonian_sparse(n, kap, h) for h in hs]
    # forward sweep (h ascending) from the best of several random starts at h_min
    fwd = [None] * nh
    best = None
    for r in range(args.n_start):
        g0 = rng.uniform(-np.pi / 2, np.pi / 2, (L, 3))
        g, e, it = hva.optimize(Hs[0], n, L, g0)
        if best is None or e < best[1]:
            best = (g, e)
    fwd[0] = best
    for j in range(1, nh):
        fwd[j] = hva.optimize(Hs[j], n, L, fwd[j - 1][0])[:2]
    # reverse sweep (h descending) from the |+>^N point g=0 plus a random start at h_max
    rev = [None] * nh
    cands = [hva.optimize(Hs[-1], n, L, 1e-2 * rng.standard_normal((L, 3)))[:2]]
    cands.append(hva.optimize(Hs[-1], n, L, rng.uniform(-np.pi / 2, np.pi / 2, (L, 3)))[:2])
    rev[-1] = min(cands, key=lambda t: t[1])
    for j in range(nh - 2, -1, -1):
        rev[j] = hva.optimize(Hs[j], n, L, rev[j + 1][0])[:2]
    # second forward pass seeded by the better branch (cheap repair of trapped points)
    keep = [fwd[j] if fwd[j][1] <= rev[j][1] else rev[j] for j in range(nh)]
    for j in range(1, nh):
        g, e, _ = hva.optimize(Hs[j], n, L, keep[j - 1][0])
        if e < keep[j][1] - 1e-12:
            keep[j] = (g, e)
    out = {k: [] for k in ["params", "E_vqe", "E_fwd", "E_rev", "E_ed", "E1_ed", "F_gs", "F_manifold",
                           "Cij", "Cr", "Sq", "mx", "qstar", "S0", "Spi2", "parity", "entropy"]}
    for j, h in enumerate(hs):
        g, e = keep[j]
        E, V, _ = ac.low_spectrum(n, kap, h, k=6)
        psi = hva.state(g, n, L)
        f0, fm, _ = vqe.manifold_fidelity(psi, E, V, window=args.manifold_window)
        gsub = ac.ground_subspace(E)
        f_sub = float(np.sum(np.abs(V[:, gsub].T.conj() @ psi) ** 2))
        o = ac.observables_state(n, psi)
        out["params"].append(g); out["E_vqe"].append(e); out["E_fwd"].append(fwd[j][1]); out["E_rev"].append(rev[j][1])
        out["E_ed"].append(E[0]); out["E1_ed"].append(E[1]); out["F_gs"].append(f_sub); out["F_manifold"].append(fm)
        for k in ["Cij", "Cr", "Sq", "mx", "qstar", "S0", "Spi2", "parity"]:
            out[k].append(o[k])
        out["entropy"].append(ac.half_chain_entropy(n, psi))
    arrays = {k: np.asarray(v) for k, v in out.items()}
    arrays["h"] = hs; arrays["kappa"] = np.array(kap)
    return arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--nk", type=int, default=31); ap.add_argument("--nh", type=int, default=31)
    ap.add_argument("--kmin", type=float, default=0.0); ap.add_argument("--kmax", type=float, default=1.0)
    ap.add_argument("--hmin", type=float, default=0.0); ap.add_argument("--hmax", type=float, default=2.0)
    ap.add_argument("--rows", default="all"); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-start", type=int, default=4); ap.add_argument("--manifold-window", type=float, default=1e-3)
    ap.add_argument("--scan", default=None)
    args = ap.parse_args()
    ks, hs = sio.grid(args.nk, args.nh, args.kmin, args.kmax, args.hmin, args.hmax)
    scan = args.scan or f"vqe_hva_N{args.n}_L{args.L}_{args.nk}x{args.nh}"
    core = {"scan": scan, "N": args.n, "L": args.L, "boundary": "pbc", "nk": args.nk, "nh": args.nh,
            "kappa_range": [args.kmin, args.kmax], "h_range": [args.hmin, args.hmax], "seed": args.seed,
            "n_start": args.n_start, "ansatz": "HVA: |+>^N then L x [exp(-i a ZZnn) exp(-i c ZZnnn) exp(-i b X)], 3L params",
            "compilation": "per ZZ bond CNOT-RZ(2angle)-CNOT, NN bonds then NNN bonds; RX(2b); 4NL CNOTs",
            "optimizer": "scipy L-BFGS-B, exact adjoint gradient, gtol 1e-10; forward+reverse warm sweeps, keep lower E, forward repair pass",
            "noise": "none (clean optimisation, p=0)", "manifold_window": args.manifold_window}
    rows = sio.parse_rows(args.rows, args.nk)
    prog = sio.Progress(len(rows), scan)
    for r in rows:
        if sio.row_done(scan, r, core):
            prog.step(f"row {r} exists, skipped"); continue
        t0 = time.time()
        arr = run_row(args, r, float(ks[r]), hs)
        meta = dict(core, row=r, kappa=float(ks[r]), runtime_s=time.time() - t0, host=socket.gethostname(),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), versions=ac.software_versions(), peak_rss_mb=sio.peak_rss_mb())
        sio.save_row(scan, r, meta, arr)
        relE = np.max((arr["E_vqe"] - arr["E_ed"]) / np.abs(arr["E_ed"]))
        prog.step(f"row {r} kappa={ks[r]:.3f} {time.time()-t0:.0f}s max relE={relE:.1e} min F_gs={arr['F_gs'].min():.3f}")


if __name__ == "__main__":
    main()
