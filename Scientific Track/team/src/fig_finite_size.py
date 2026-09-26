"""Finite-size study (clean ED, PBC N=8,10,12) and floating-phase evidence assessment."""
import json
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt
import annni_core as ac
import phase_analysis as pa
import scan_io as sio

FIG = sio.RESULTS / "figures"; AN = sio.RESULTS / "analysis"
COL = {8: "#2a78d6", 10: "#eb6834", 12: "#1baf7a"}


def main():
    cal = json.loads((AN / "calibration_ed_N8.json").read_text())
    T = cal["T_order"]
    # ---- coarse 31x31 maps for N=8,10,12
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    grids = {}
    for ax, n in zip(axs, (8, 10, 12)):
        _, d, _ = pa.load_scan(f"ed_N{n}_pbc_31x31", 31)
        OF, OA = pa.orders(d["Sq"], n)
        if n % 4:  # N=10: no pi/2 momentum; use max over q in (0,pi) excluding 0 as modulated-order proxy
            qs = 2 * np.pi * np.arange(n) / n
            k = np.argmin(np.abs(qs[: n // 2 + 1] - np.pi / 2))
            OA = (d["Sq"][..., k] - 1) / (n / 2 - 1)
        lab = pa.classify(OF, OA, d["mx"], T, cal["T_mx"])
        grids[n] = (d, OF, OA, lab)
        fs.phase_map(ax, d["kappa"], d["h"], lab, None, f"ED N={n} PBC" + (" (q nearest π/2 = 2π·2/10)" if n % 4 else ""))
        fs.overlays(ax, legend=(n == 8), pe=False, lw=1.0)
    fs.phase_legend(fig, ncol=4); fig.subplots_adjust(bottom=0.22, wspace=0.06)
    fig.savefig(FIG / "fig_finite_size_maps.png"); plt.close(fig)
    # ---- fine cuts: boundaries from threshold, derivative, fidelity susceptibility
    out = {}
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for n in (8, 10, 12):
        try:
            _, d, _ = pa.load_scan(f"ed_N{n}_pbc_21x401", 21)
        except Exception as e:
            print("fine scan missing", n, e); continue
        K, H = d["kappa"], d["h"]
        OF, OA = pa.orders(d["Sq"], n)
        if n % 4:
            qs = 2 * np.pi * np.arange(n) / n
            k = np.argmin(np.abs(qs[: n // 2 + 1] - np.pi / 2))
            OA = (d["Sq"][..., k] - 1) / (n / 2 - 1)
        b = pa.boundaries(K, H, OF, OA, T)
        chi = d["chiF_h_per_site"].copy()
        # fidelity between ground clusters of different dimension (tunnelling doublet crossing the
        # degeneracy tolerance) is not a transition signal: exclude those pairs
        deg = d["deg"]
        chi[:, :-1][deg[:, :-1] != deg[:, 1:]] = np.nan
        hF = np.array([pa.fidelity_peak(H, chi[i]) for i in range(len(K))])
        out[n] = {"kappa": K.tolist(), "h_T": b["h_T"].tolist(), "h_D": b["h_D"].tolist(), "h_chiF": hF.tolist(),
                  "chiF_max_per_site": np.nanmax(chi[:, :-1], axis=1).tolist()}
        ax.plot(K, b["h_D"], "o-", color=COL[n], ms=3, lw=1.2, label=f"N={n} max|dO/dh|")
        ax.plot(K, hF, "x:", color=COL[n], ms=4, lw=0.8, label=f"N={n} χ_F peak")
    fs.overlays(ax, legend=False, pe=False)
    ax.legend(fontsize=9.1, frameon=False, ncol=2, loc="upper center")
    ax.set_title("Clean ED finite-ring boundary estimators (δh=0.005)"); ax.set_xlabel("κ"); ax.set_ylabel("h")
    fig.savefig(FIG / "fig_finite_size_boundaries.png"); plt.close(fig)
    ref = {"kappa": out.get(8, {}).get("kappa"), "h_ising": None}
    if 8 in out:
        k = np.array(out[8]["kappa"])
        ref = {"h_I": ac.h_ising(k).tolist(), "h_KT": ac.h_kt_upper(k).tolist(), "h_PT": ac.h_pt_lower(k).tolist()}
    pa.save_json(AN / "finite_size_boundaries.json", {"per_N": out, "reference_curves": ref, "threshold": T})
    # ---- floating-phase evidence: kappa=0.75 and 0.9 cuts, q*, C(r), same-parity gap, entropy vs N
    fig, axs = plt.subplots(2, 3, figsize=(12, 5.6))
    ev = {}
    for row, kv in enumerate((0.75, 0.9)):
        for n in (8, 10, 12):
            try:
                _, d, _ = pa.load_scan(f"ed_N{n}_pbc_21x401", 21)
            except Exception:
                continue
            i = int(np.argmin(abs(d["kappa"] - kv))); H = d["h"]
            qi = np.argmax(d["Sq"][i][:, : n // 2 + 1], axis=1)
            q = 2 * np.pi * qi / n / np.pi
            axs[row, 0].plot(H, q, color=COL[n], lw=1.2, label=f"N={n}")
            E, par = d["E"][i], d["eig_parity"][i]
            # lowest gap above ground to a state of different momentum/parity: use E1-E0 (cluster excluded)
            gap = E[:, 1] - E[:, 0]
            axs[row, 1].semilogy(H, np.clip(gap, 1e-12, None), color=COL[n], lw=1.2, label=f"N={n}")
            axs[row, 2].plot(H, d["entropy"][i], color=COL[n], lw=1.2, label=f"N={n}")
            # level crossings = jumps in q* (ground-state momentum/pattern changes)
            jumps = H[1:][np.diff(qi) != 0]
            ev[f"kappa{kv}_N{n}_qstar_jumps_h"] = jumps.tolist()
            ev[f"kappa{kv}_N{n}_min_gap"] = float(np.min(gap[H > 0.05]))
        axs[row, 0].set_title(f"κ={kv}: dominant q*/π"); axs[row, 1].set_title(f"κ={kv}: E1−E0 (log)")
        axs[row, 2].set_title(f"κ={kv}: half-chain entropy S_N/2 (pure GS)")
        for a in axs[row]:
            a.axvline(float(ac.h_pt_lower(kv)), color="k", ls="--", lw=0.8)
            a.axvline(float(ac.h_kt_upper(kv)), color="k", ls=":", lw=0.8)
            a.set_xlim(0, 1.5)
        axs[row, 0].legend(fontsize=10.2, frameon=False)
    for a in axs[1]:
        a.set_xlabel("h")
    fig.suptitle("Floating-phase test cuts (dashed: lower PT fit, dotted: upper KT fit)", fontsize=13.0)
    fig.tight_layout(); fig.savefig(FIG / "fig_floating_cuts.png"); plt.close(fig)
    pa.save_json(AN / "floating_evidence.json", ev)
    print(json.dumps({n: {k: np.round(v, 3).tolist() for k, v in o.items() if k != "kappa"} for n, o in out.items()}, indent=0)[:3000])
    print(json.dumps(ev, indent=0)[:2500])


if __name__ == "__main__":
    main()
