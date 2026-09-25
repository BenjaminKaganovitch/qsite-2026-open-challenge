"""Circuit (HVA, N=8, L=4) diagrams at p=0, 0.01, 0.05 (fixed parameters), preparation-error maps,
boundary-shift tables with uncertainty budget, first-order-response comparison [S28]."""
import json, csv
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt
import phase_analysis as pa
import scan_io as sio

FIG = sio.RESULTS / "figures"; AN = sio.RESULTS / "analysis"
N, L = 8, 4
PS = (0.0, 0.01, 0.05)
NCH = 4 * N * L


def load_all(nk=31):
    cal = json.loads((AN / "calibration_ed_N8.json").read_text())
    _, ed, _ = pa.load_scan("ed_N8_pbc_31x31", nk)
    _, vq, _ = pa.load_scan(f"vqe_hva_N{N}_L{L}_31x31", nk)
    _, nz, _ = pa.load_scan(f"noisy_hva_N{N}_L{L}_31x31", nk)
    try:
        _, ex, _ = pa.load_scan(f"noisy_extras_hva_N{N}_L{L}_31x31", nk)
    except Exception as e:  # extras optional
        print("extras not available:", e); ex = None
    return cal, ed, vq, nz, ex


def prep_unreliable(vq, fmin=0.9):
    return np.fmax(vq["F_gs"], vq["F_manifold"]) < fmin


def main():
    cal, ed, vq, nz, ex = load_all()
    T, Tx = cal["T_order"], cal["T_mx"]
    K, H = ed["kappa"], ed["h"]
    OFe, OAe = pa.orders(ed["Sq"], N)
    lab_ed = pa.classify(OFe, OAe, ed["mx"], T, Tx)
    bad = prep_unreliable(vq)
    res = {}
    for p in PS:
        OF, OA = pa.orders(nz[f"p{p}_Sq"], N)
        mx = nz[f"p{p}_mx"]
        res[p] = dict(OF=OF, OA=OA, mx=mx, lab=pa.classify(OF, OA, mx, T, Tx), amb=pa.ambiguity(OF, OA, mx, T, Tx))
        OFv, OAv = pa.orders(nz[f"p{p}_Sq_ver"], N)
        res[p].update(OFv=OFv, OAv=OAv, mxv=nz[f"p{p}_mx_ver"], labv=pa.classify(OFv, OAv, nz[f"p{p}_mx_ver"], T, Tx))
    for p in PS:
        res[p]["labR"] = pa.classify_ratio(res[p]["OF"], res[p]["OA"], res[p]["mx"])
    np.savez_compressed(AN / "labels_circuit_N8_L4.npz", **{f"labR_p{p}": res[p]["labR"] for p in PS}, kappa=K, h=H, prep_unreliable=bad, lab_ed=lab_ed,
                        **{f"lab_p{p}": res[p]["lab"] for p in PS}, **{f"amb_p{p}": res[p]["amb"] for p in PS},
                        **{f"labver_p{p}": res[p]["labv"] for p in PS})
    # ---------------- required diagrams (one file each) + combined panel
    KK, HH = np.meshgrid(K, H, indexing="ij")
    lab_ed_R = pa.classify_ratio(OFe, OAe, ed["mx"])
    for p in PS:
        res[p]["labR"] = pa.classify_ratio(res[p]["OF"], res[p]["OA"], res[p]["mx"])
        fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
        fs.phase_map(axs[0], K, H, res[p]["lab"], res[p]["amb"], f"(a) fixed clean-ED thresholds, p={p}")
        fs.phase_map(axs[1], K, H, res[p]["labR"], None, f"(b) relative dominance argmax(O_F,O_A,m_x²), p={p}")
        for ax in axs:
            ax.scatter(KK[bad], HH[bad], s=7, marker="s", facecolors="none", edgecolors="#6b1d8a", lw=0.6)
            fs.overlays(ax, pe=False, legend=(ax is axs[0]))
        sig = np.fmax(np.fmax(res[p]["OF"], res[p]["OA"]), np.square(res[p]["mx"]))
        weak = sig < 0.05
        if weak.any():
            axs[1].scatter(KK[weak], HH[weak], s=2, marker=".", color="#0b0b0b", lw=0)
        axs[1].set_ylabel("")
        fig.suptitle(f"HVA circuit N=8 PBC, L=4 ({NCH} CNOTs, {NCH} target channels), p={p}", fontsize=9.5)
        fs.phase_legend(fig, ncol=4, y=0.06)
        fig.text(0.5, 0.01, "Fixed clean-optimised parameters for all p. Purple squares: clean preparation fidelity < 0.9. "
                 "x: label changes for thresholds ±0.1. Black dots in (b): largest signal < 0.05.", ha="center", fontsize=6.5)
        fig.subplots_adjust(bottom=0.27, wspace=0.08)
        fig.savefig(FIG / f"phase_diagram_p{p}.png"); fig.savefig(FIG / f"phase_diagram_p{p}.pdf"); plt.close(fig)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    lab_ed_amb = pa.ambiguity(OFe, OAe, ed["mx"], T, Tx)
    fs.phase_map(axs[0], K, H, lab_ed, lab_ed_amb, "Exact ground states (ED), independent reference")
    for ax, p in zip(axs[1:], PS):
        fs.phase_map(ax, K, H, res[p]["lab"], res[p]["amb"], f"HVA circuit, p={p}")
        ax.scatter(KK[bad], HH[bad], s=6, marker="s", facecolors="none", edgecolors="#6b1d8a", lw=0.5)
    for i, ax in enumerate(axs):
        fs.overlays(ax, legend=(i == 0), pe=False, lw=1.0)
        if i:
            ax.set_ylabel("")
    fs.phase_legend(fig, ncol=4)
    fig.subplots_adjust(bottom=0.24, wspace=0.08)
    fig.savefig(FIG / "fig_phase_diagrams_all.png"); plt.close(fig)
    # ---------------- preparation quality maps
    relE = (vq["E_vqe"] - vq["E_ed"]) / np.abs(vq["E_ed"])
    Fbest = np.fmax(vq["F_gs"], vq["F_manifold"])
    dS = np.max(np.abs(nz["p0.0_Sq"] - ed["Sq"]), axis=-1)
    dis = (res[0.0]["lab"] != lab_ed)
    ext = [K[0], K[-1], H[0], H[-1]]
    fig, axs = plt.subplots(1, 4, figsize=(13, 2.9))
    for ax, arr, t, kw in zip(axs, [np.log10(np.clip(relE, 1e-9, None)), 1 - Fbest, dS, dis.astype(float)],
                              ["log10 relative energy error", "1 − ground-subspace/manifold fidelity", "max_q |S_circ(q) − S_ED(q)|", "label disagreement (circuit p=0 vs ED)"],
                              [dict(cmap="magma_r", vmin=-7, vmax=-1), dict(cmap="magma_r", vmin=0, vmax=1), dict(cmap="magma_r", vmin=0, vmax=1), dict(cmap="Greys", vmin=0, vmax=1)]):
        im = ax.imshow(arr.T, origin="lower", extent=ext, aspect="auto", **kw)
        fs.overlays(ax, lw=0.7, legend=False, pe=False); ax.set_title(t, fontsize=8); ax.set_xlabel("κ")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axs[0].set_ylabel("h")
    fig.savefig(FIG / "fig_circuit_vs_ed.png"); plt.close(fig)
    prep = {"max_relE": float(relE.max()), "median_relE": float(np.median(relE)),
            "frac_F_below_0.9": float(bad.mean()), "median_1mF": float(np.median(1 - Fbest)),
            "max_dSq": float(dS.max()), "median_dSq": float(np.median(dS)),
            "label_disagreement_count": int(dis.sum()), "label_disagreement_frac": float(dis.mean()),
            "p0_mixed_vs_pure_note": "p=0 row uses exact pure circuit state; default.mixed p=0 agreement validated to 3e-14 in circuit_validation.json",
            "branch_disagreement_max": float(np.max(np.abs(vq["E_fwd"] - vq["E_rev"])))}
    # ---------------- boundaries and shifts
    rows = []
    ph_clean = pa.boundaries(K, H, res[0.0]["OF"], res[0.0]["OA"], T)["phase"]
    bED = pa.boundaries(K, H, OFe, OAe, T)
    out = {}
    for p in PS:
        OF, OA = res[p]["OF"], res[p]["OA"]
        hT, hTp, hTlo, hThi, hD = [], [], [], [], []
        for i in range(len(K)):
            ph = int(ph_clean[i])
            if ph < 0:
                hT.append(np.nan); hTp.append(np.nan); hTlo.append(np.nan); hThi.append(np.nan); hD.append(np.nan); continue
            y = OF[i] if ph == 0 else OA[i]
            hT.append(pa._crossing(H, y, T)); hTp.append(pa._crossing(H, y, T, "pchip"))
            hTlo.append(pa._crossing(H, y, T + 0.1)); hThi.append(pa._crossing(H, y, T - 0.1))
            hD.append(pa._deriv_peak(H, y))
        out[p] = {k: np.array(v) for k, v in dict(h_T=hT, h_T_pchip=hTp, h_T_lo=hTlo, h_T_hi=hThi, h_D=hD).items()}
    dh = H[1] - H[0]
    for i, kv in enumerate(K):
        if ph_clean[i] < 0:
            continue
        r = {"kappa": round(float(kv), 4), "order": ["ferro", "antiphase"][int(ph_clean[i])],
             "prep_unreliable_frac_row": float(bad[i].mean())}
        r["h_T_ED"] = bED["h_T"][i]
        for p in PS:
            r[f"h_T_p{p}"] = out[p]["h_T"][i]; r[f"h_D_p{p}"] = out[p]["h_D"][i]
        r["prep_bias_h_T"] = out[0.0]["h_T"][i] - bED["h_T"][i]
        for p in PS:
            lr = res[p]["labR"][i]
            jj = int(np.argmax(lr == 2)) if np.any(lr == 2) else -1
            r[f"h_R_p{p}"] = float(0.5 * (H[jj - 1] + H[jj])) if jj > 0 else np.nan
        for p in (0.01, 0.05):
            r[f"dh_R_p{p}"] = r[f"h_R_p{p}"] - r["h_R_p0.0"]
        for p in (0.01, 0.05):
            r[f"dh_T_p{p}"] = out[p]["h_T"][i] - out[0.0]["h_T"][i]
            r[f"dh_D_p{p}"] = out[p]["h_D"][i] - out[0.0]["h_D"][i]
            # uncertainty budget for dh_T
            r[f"u_interp_p{p}"] = abs((out[p]["h_T_pchip"][i] - out[0.0]["h_T_pchip"][i]) - r[f"dh_T_p{p}"])
            thr = [out[p]["h_T_lo"][i] - out[0.0]["h_T_lo"][i], out[p]["h_T_hi"][i] - out[0.0]["h_T_hi"][i]]
            r[f"u_threshold_p{p}"] = float(np.nanmax(np.abs(np.array(thr) - r[f"dh_T_p{p}"]))) if np.any(np.isfinite(thr)) else np.nan
            r[f"u_grid_D_p{p}"] = dh / 2
        rows.append(r)
    with open(AN / "boundary_shifts_circuit_N8_L4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    pa.save_json(AN / "prep_quality_N8_L4.json", prep)
    # ---------------- line cuts
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.2), sharex=True)
    for j, kv in enumerate((0.2, 0.3, 0.7, 0.9)):
        i = int(np.argmin(abs(K - kv)))
        ph = int(ph_clean[i]); key = "OF" if ph == 0 else "OA"
        ye = OFe[i] if ph == 0 else OAe[i]
        axs[0, j].plot(H, ye, color="k", lw=1, label="ED")
        for p, c in zip(PS, ["#2a78d6", "#eb6834", "#1baf7a"]):
            axs[0, j].plot(H, res[p][key][i], color=c, marker="o", ms=2.5, lw=1.2, label=f"circuit p={p}")
            if p > 0:
                axs[0, j].plot(H, res[p][key + "v"][i], color=c, ls="--", lw=1, label=f"parity-verified p={p}")
            axs[1, j].plot(H, res[p]["mx"][i], color=c, marker="o", ms=2.5, lw=1.2)
        axs[1, j].plot(H, ed["mx"][i], color="k", lw=1)
        axs[0, j].axhline(T, color="#999", lw=0.7, ls=":")
        axs[0, j].set_title(f"κ={K[i]:.2f}: {'O_F' if ph == 0 else 'O_A'}")
        axs[1, j].set_title(f"κ={K[i]:.2f}: m_x"); axs[1, j].set_xlabel("h")
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig_noise_cuts.png"); plt.close(fig)
    # ---------------- contraction map vs terminal benchmark
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for ax, p in zip(axs, (0.01, 0.05)):
        qi = np.argmax(nz["p0.0_Sq"][..., : N // 2 + 1], axis=-1)
        s0 = np.take_along_axis(nz["p0.0_Sq"], qi[..., None], -1)[..., 0]
        sp = np.take_along_axis(nz[f"p{p}_Sq"], qi[..., None], -1)[..., 0]
        R = np.where(s0 - 1 > 0.2, (sp - 1) / (s0 - 1), np.nan)
        im = ax.imshow(R.T, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        a = 1 - 4 * p / 3
        ax.set_title(f"(S_p(q*)−1)/(S_0(q*)−1), p={p}\nterminal ref. a²={a**2:.3f} per layer of hits", fontsize=8)
        fs.overlays(ax, lw=0.7, legend=False, pe=False); ax.set_xlabel("κ")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axs[0].set_ylabel("h"); fig.savefig(FIG / "fig_contraction.png"); plt.close(fig)
    summ = {}
    for p in (0.01, 0.05):
        qi = np.argmax(nz["p0.0_Sq"][..., : N // 2 + 1], axis=-1)
        s0 = np.take_along_axis(nz["p0.0_Sq"], qi[..., None], -1)[..., 0]
        sp = np.take_along_axis(nz[f"p{p}_Sq"], qi[..., None], -1)[..., 0]
        m = s0 - 1 > 0.2
        R = (sp - 1)[m] / (s0 - 1)[m]
        summ[f"contraction_p{p}"] = {"median": float(np.median(R)), "p10": float(np.percentile(R, 10)), "p90": float(np.percentile(R, 90))}
        for ph_i, name in ((0, "ferro"), (1, "antiphase"), (2, "paramagnet")):
            mm = lab_ed == ph_i
            summ[f"frac_ED_{name}_kept_p{p}"] = float(np.mean(res[p]["lab"][mm] == ph_i))
            summ[f"frac_ED_{name}_kept_parityver_p{p}"] = float(np.mean(res[p]["labv"][mm] == ph_i))
        summ[f"accept_parity_p{p}_median"] = float(np.median(nz[f"p{p}_accept"]))
        summ[f"purity_p{p}_median"] = float(np.median(nz[f"p{p}_purity"]))
        summ[f"mx_ratio_p{p}_median_para"] = float(np.median((nz[f"p{p}_mx"] / nz["p0.0_mx"])[lab_ed == 2]))
    for ph_i, name in ((0, "ferro"), (1, "antiphase"), (2, "paramagnet")):
        summ[f"frac_ED_{name}_kept_p0.0"] = float(np.mean(res[0.0]["lab"][lab_ed == ph_i] == ph_i))
    summ["ratio_label_counts"] = {str(p): [int((res[p]["labR"] == i).sum()) for i in range(3)] for p in PS}
    summ["ratio_label_agreement_with_ED_ratio"] = {str(p): float(np.mean(res[p]["labR"] == lab_ed_R)) for p in PS}
    summ["label_counts"] = {str(p): [int((res[p]["lab"] == i).sum()) for i in range(4)] for p in PS}
    summ["label_counts_parity_verified"] = {str(p): [int((res[p]["labv"] == i).sum()) for i in range(4)] for p in PS}
    pa.save_json(AN / "noise_summary_N8_L4.json", summ)
    print(json.dumps(prep, indent=1)); print(json.dumps(summ, indent=1))
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("kappa", "order", "h_T_ED", "h_T_p0.0", "h_T_p0.01", "h_T_p0.05", "h_D_p0.0", "h_D_p0.01", "h_D_p0.05", "prep_unreliable_frac_row")})


if __name__ == "__main__":
    main()
