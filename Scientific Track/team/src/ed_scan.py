"""Clean sparse-ED reference scan (CLI). One kappa row = one complete, warm-started h sweep.

Example:  python team/src/ed_scan.py --n 8 --rows 0:31
Writes    team/results/ed_N8_pbc_31x31/row_<iii>.npz (resumable; existing valid rows skipped)."""
from scan_io import set_single_thread
set_single_thread()
import argparse, json, socket, time
import numpy as np
import annni_core as ac
import scan_io as sio


def run_row(args, row, kap, hs):
    n, k = args.n, args.k
    nh = len(hs)
    out = {key: [] for key in ["E", "deg", "eig_parity", "Cr", "Sq", "Cij", "mx", "qstar", "S0", "Spi2",
                               "gs_parity", "entropy", "residual", "orth", "dE_cold"]}
    subspaces, gs_vecs = [], []
    v0 = None
    for j, h in enumerate(hs):
        E, V, info = ac.low_spectrum(n, kap, h, k=k, periodic=not args.obc, v0=v0, seed=args.seed + 7919 * row + j)
        # independent cold start (different random vector, no warm start) to detect branch/sector trapping
        Ec, Vc, _ = ac.low_spectrum(n, kap, h, k=k, periodic=not args.obc, v0=None, seed=args.seed + 104729 + 7919 * row + j)
        if Ec[0] < E[0] - 1e-9:
            E, V = Ec, Vc
        g = ac.ground_subspace(E, tol=args.degtol)
        Vg = V[:, g]
        obs = ac.observables_subspace(n, Vg)
        out["E"].append(E); out["deg"].append(len(g))
        out["eig_parity"].append([float(V[:, a] @ V[::-1, a]) for a in range(k)])
        for key in ["Cr", "Sq", "Cij", "mx", "qstar", "S0", "Spi2"]:
            out[key].append(obs[key])
        out["gs_parity"].append(obs["parity"])
        out["entropy"].append(ac.half_chain_entropy(n, V[:, 0]) if len(g) == 1 else np.nan)
        out["residual"].append(info["residual"]); out["orth"].append(info["orthogonality"])
        out["dE_cold"].append(float(abs(Ec[0] - E[0])))
        subspaces.append(Vg); gs_vecs.append(V[:, 0])
        v0 = V[:, 0]
    F = [ac.subspace_fidelity(subspaces[j], subspaces[j + 1]) for j in range(nh - 1)] + [np.nan]
    dh = np.diff(hs).tolist() + [np.nan]
    arrays = {kk: np.asarray(v) for kk, v in out.items()}
    arrays["F_next_h"] = np.asarray(F)
    with np.errstate(divide="ignore", invalid="ignore"):
        arrays["chiF_h_per_site"] = -np.log(np.clip(arrays["F_next_h"], 1e-300, 1)) / np.asarray(dh) ** 2 / n
    arrays["h"] = hs; arrays["kappa"] = np.array(kap)
    if args.save_states:
        arrays["gs_vec"] = np.asarray(gs_vecs)
    return arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--nk", type=int, default=31); ap.add_argument("--nh", type=int, default=31)
    ap.add_argument("--kmin", type=float, default=0.0); ap.add_argument("--kmax", type=float, default=1.0)
    ap.add_argument("--hmin", type=float, default=0.0); ap.add_argument("--hmax", type=float, default=2.0)
    ap.add_argument("--rows", default="all", help="kappa-row shard, e.g. 0:8 or 3,5,9")
    ap.add_argument("--k", type=int, default=6, help="number of low eigenpairs")
    ap.add_argument("--degtol", type=float, default=1e-8)
    ap.add_argument("--obc", action="store_true", help="open boundaries (default periodic)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--scan", default=None)
    ap.add_argument("--save-states", type=int, default=None, help="store ground vectors (default: N<=10)")
    args = ap.parse_args()
    if args.save_states is None:
        args.save_states = int(args.n <= 10)
    ks, hs = sio.grid(args.nk, args.nh, args.kmin, args.kmax, args.hmin, args.hmax)
    bc = "obc" if args.obc else "pbc"
    scan = args.scan or f"ed_N{args.n}_{bc}_{args.nk}x{args.nh}"
    rows = sio.parse_rows(args.rows, args.nk)
    core = {"scan": scan, "N": args.n, "boundary": bc, "nk": args.nk, "nh": args.nh,
            "kappa_range": [args.kmin, args.kmax], "h_range": [args.hmin, args.hmax], "k_eig": args.k,
            "degtol": args.degtol, "seed": args.seed, "method": "scipy.sparse.linalg.eigsh(which=SA), warm start along h + independent cold start",
            "hamiltonian": "-sum ZZ(i,i+1) + kappa sum ZZ(i,i+2) - h sum X, Pauli, J1=1",
            "normalization": "C(r)=(1/N)sum_i<ZiZi+r>; S(q)=(1/N)sum_ij e^{iq(i-j)}<ZiZj> incl. Cii=1; mx=(1/N)sum<Xi>",
            "degenerate_ground_space": "observables = equal mixture over eigenvalues within degtol*max(1,|E0|)",
            "fidelity": "F_next_h = mean principal squared subspace fidelity between h_j and h_j+1; chiF=-lnF/dh^2/N",
            "save_states": args.save_states}
    prog = sio.Progress(len(rows), scan)
    for r in rows:
        if sio.row_done(scan, r, core):
            prog.step(f"row {r} exists, skipped"); continue
        t0 = time.time()
        arrays = run_row(args, r, float(ks[r]), hs)
        meta = dict(core, row=r, kappa=float(ks[r]), runtime_s=time.time() - t0, host=socket.gethostname(),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), versions=ac.software_versions(),
                    peak_rss_mb=sio.peak_rss_mb())
        sio.save_row(scan, r, meta, arrays)
        prog.step(f"row {r} kappa={ks[r]:.4f} {time.time()-t0:.1f}s maxres={arrays['residual'].max():.1e} maxdEcold={arrays['dE_cold'].max():.1e}")


if __name__ == "__main__":
    main()
