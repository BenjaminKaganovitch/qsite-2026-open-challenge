"""Clean ED figures + calibration + ED boundary table (regenerated from cached arrays)."""
import json, sys
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt
import annni_core as ac
import phase_analysis as pa
import scan_io as sio

FIG = sio.RESULTS / "figures"; FIG.mkdir(parents=True, exist_ok=True)
AN = sio.RESULTS / "analysis"; AN.mkdir(parents=True, exist_ok=True)


def ed_bundle(n, nk=31, nh=31):
    metas, d, _ = pa.load_scan(f"ed_N{n}_pbc_{nk}x{nh}", nk)
    OF, OA = pa.orders(d["Sq"], n)
    return d, OF, OA


def main():
    d, OF, OA = ed_bundle(8)
    K, H = d["kappa"], d["h"]
    KK, HH = np.meshgrid(K, H, indexing="ij")
    cal = pa.calibrate(OF, OA, d["mx"], KK, HH)
    T, Tx = cal["T_order"], cal["T_mx"]
    lab = pa.classify(OF, OA, d["mx"], T, Tx)
    amb = pa.ambiguity(OF, OA, d["mx"], T, Tx)
    mod, qi = pa.modulated(d["Sq"], 8)
    pa.save_json(AN / "calibration_ed_N8.json", cal)
    np.savez_compressed(AN / "labels_ed_N8.npz", kappa=K, h=H, labels=lab, ambiguous=amb, modulated=mod, qstar_idx=qi, OF=OF, OA=OA, mx=d["mx"])
    # --- Figure: clean ED phase diagram
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    fs.phase_map(ax, K, H, lab, amb, "Clean ED reference, N=8 PBC, 31×31", mod=None)
    fs.overlays(ax)
    fs.phase_legend(fig, loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(FIG / "fig_ed_phase_N8.png"); plt.close(fig)
    # --- raw observable maps
    ext = [K[0], K[-1], H[0], H[-1]]
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.9))
    for ax, arr, t, cm, vr in zip(axs, [OF, OA, d["mx"], qi * 2 * np.pi / 8 / np.pi],
                                  ["O_F=(S(0)-1)/(N-1)", "O_A=(S(π/2)-1)/(N/2-1)", "m_x", "q*/π"],
                                  ["Blues", "Oranges", "Greens", "Purples"], [(0, 1), (0, 1), (0, 1), (0, 1)]):
        im = ax.imshow(arr.T, origin="lower", extent=ext, aspect="auto", cmap=cm, vmin=vr[0], vmax=vr[1])
        fs.overlays(ax, lw=0.8, legend=False, pe=False); ax.set_title(t); ax.set_xlabel("κ")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axs[0].set_ylabel("h")
    fig.suptitle("Clean ED raw diagnostics, N=8 PBC (overlays: Ising solid, PT dashed, KT dotted)", fontsize=13.0)
    fig.savefig(FIG / "fig_ed_observables_N8.png"); plt.close(fig)
    # --- threshold sensitivity
    bd = {}
    for t in (T - 0.15, T - 0.1, T, T + 0.1, T + 0.15):
        b = pa.boundaries(K, H, OF, OA, t)
        bd[f"{t:.3f}"] = b["h_T"]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for i, (t, hb) in enumerate(bd.items()):
        ax.plot(K, hb, marker="o", ms=2.5, lw=1, label=f"T={t}", color=plt.cm.viridis(i / 4))
    fs.overlays(ax, legend=False)
    ax.legend(fontsize=9.4, frameon=False, loc="upper center", ncol=3)
    ax.set_title("ED N=8: order-threshold crossing h_T(κ) vs threshold"); ax.set_xlabel("κ"); ax.set_ylabel("h")
    fig.savefig(FIG / "fig_ed_threshold_sensitivity_N8.png"); plt.close(fig)
    # --- ED fidelity / gap / entropy on the grid for three cuts
    fig, axs = plt.subplots(1, 3, figsize=(11, 2.9))
    for kv, c in zip((0.2, 0.5, 0.8), fs.PHASE_COLORS[:3]):
        i = int(np.argmin(abs(K - kv)))
        axs[0].plot(0.5 * (H[:-1] + H[1:]), d["chiF_h_per_site"][i, :-1], color=c, marker="o", ms=2.5, label=f"κ={K[i]:.2f}")
        E = d["E"][i]; par = d["eig_parity"][i]
        gap = E[:, 1] - E[:, 0]
        # lowest excitation in the same parity sector as E0 (tunnel-splitting-safe gap)
        gs = np.sign(par[:, 0]); same = np.array([next((E[j, a] - E[j, 0] for a in range(1, E.shape[1]) if np.sign(par[j, a]) == gs[j] and abs(par[j, a]) > 0.5), np.nan) for j in range(len(H))])
        axs[1].semilogy(H, np.clip(gap, 1e-12, None), color=c, ls=":", lw=1)
        axs[1].semilogy(H, same, color=c, marker="o", ms=2.5, label=f"κ={K[i]:.2f}")
        axs[2].plot(H, d["entropy"][i], color=c, marker="o", ms=2.5, label=f"κ={K[i]:.2f}")
    axs[0].set_title("fidelity susceptibility χ_F/N (δh=1/15)"); axs[1].set_title("gap: E1−E0 (dotted), same-parity gap (solid)")
    axs[2].set_title("half-chain entanglement entropy S_{N/2}")
    for ax in axs:
        ax.set_xlabel("h"); ax.legend(fontsize=10.2, frameon=False)
    fig.savefig(FIG / "fig_ed_cuts_N8.png"); plt.close(fig)
    print(json.dumps(cal, indent=1))
    frac = {pa.LABELS[i]: int((lab == i).sum()) for i in range(4)}
    print("label counts", frac, "ambiguous", int(amb.sum()))


if __name__ == "__main__":
    main()
