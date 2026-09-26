"""ZNE on correlators (fixed circuit parameters, p=0 truth held out), parity verification, and the
exact first-order noise response vs measured boundary shifts [S19, S20, S21, S28]."""
import json, csv
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt
import annni_core as ac
import phase_analysis as pa
import scan_io as sio
from fig_noise import load_all, N, L, PS

FIG = sio.RESULTS / "figures"; AN = sio.RESULTS / "analysis"


def sq_from_cij(C):
    """Rebuild S(q) from (possibly mitigated) correlator matrices; diagonal reset to exactly 1."""
    C = C.copy()
    idx = np.arange(N)
    C[..., idx, idx] = 1.0
    q = 2 * np.pi * np.arange(N) / N
    ph = np.cos(q[:, None, None] * (idx[None, :, None] - idx[None, None, :]))
    return np.einsum("qij,...ij->...q", ph, C) / N


def richardson(vals, c):
    """Weights with sum g = 1 and sum g c^k = 0 (k=1..m-1) [S19]."""
    c = np.asarray(c, float); m = len(c)
    A = np.vstack([c**k for k in range(m)])
    g = np.linalg.solve(A, np.eye(m)[0])
    return sum(gi * v for gi, v in zip(g, vals)), g


def main():
    cal, ed, vq, nz, ex = load_all()
    assert ex is not None, "extras scan required"
    T, Tx = cal["T_order"], cal["T_mx"]
    K, H = ed["kappa"], ed["h"]
    OFe, OAe = pa.orders(ed["Sq"], N)
    lab_ed = pa.classify(OFe, OAe, ed["mx"], T, Tx)
    C0, x0 = nz["p0.0_Cij"], nz["p0.0_mx"]
    S0 = sq_from_cij(C0)
    series = {
        0.01: {"lin": [(1, nz["p0.01_Cij"], nz["p0.01_mx"]), (2, ex["lin_0.02_Cij"], ex["lin_0.02_mx"]), (3, ex["lin_0.03_Cij"], ex["lin_0.03_mx"])]},
        0.05: {"lin": [(1, nz["p0.05_Cij"], nz["p0.05_mx"]), (2, ex["lin_0.1_Cij"], ex["lin_0.1_mx"]), (3, ex["lin_0.15_Cij"], ex["lin_0.15_mx"])],
               "comp": [(1, nz["p0.05_Cij"], nz["p0.05_mx"]), (2, ex["comp_0.05_l2_Cij"], ex["comp_0.05_l2_mx"]), (3, ex["comp_0.05_l3_Cij"], ex["comp_0.05_l3_mx"])]},
    }
    report, mitig = {}, {}
    for p, dd in series.items():
        for path, pts in dd.items():
            for order_ in (2, 3):
                use = pts[:order_]
                Cm, g = richardson([u[1] for u in use], [u[0] for u in use])
                xm, _ = richardson([u[2] for u in use], [u[0] for u in use])
                Sm = sq_from_cij(Cm)
                OF, OA = pa.orders(Sm, N)
                lab = pa.classify(OF, OA, xm, T, Tx)
                key = f"p{p}_{path}_R{order_}"
                off = ~np.eye(N, dtype=bool)
                err_raw = np.abs(pts[0][1] - C0)[..., off].mean(axis=-1)
                err_m = np.abs(Cm - C0)[..., off].mean(axis=-1)
                mitig[key] = dict(OF=OF, OA=OA, mx=xm, lab=lab, Sq=Sm)
                report[key] = {"weights": g.tolist(), "median_abs_Cij_err_raw": float(np.median(err_raw)),
                               "median_abs_Cij_err_mitigated": float(np.median(err_m)),
                               "median_abs_mx_err_raw": float(np.median(np.abs(pts[0][2] - x0))),
                               "median_abs_mx_err_mitigated": float(np.median(np.abs(xm - x0))),
                               "label_agreement_with_clean_circuit_raw": float(np.mean(pa.classify(*pa.orders(sq_from_cij(pts[0][1]), N), pts[0][2], T, Tx) == pa.classify(*pa.orders(S0, N), x0, T, Tx))),
                               "label_agreement_with_clean_circuit_mitigated": float(np.mean(lab == pa.classify(*pa.orders(S0, N), x0, T, Tx))),
                               "variance_amplification_sum_g2": float(np.sum(np.square(g)))}
    # parity verification
    for p in (0.01, 0.05):
        off = ~np.eye(N, dtype=bool)
        report[f"p{p}_parity_verified"] = {
            "median_abs_Cij_err_raw": float(np.median(np.abs(nz[f"p{p}_Cij"] - C0)[..., off].mean(-1))),
            "median_abs_Cij_err_verified": float(np.median(np.abs(nz[f"p{p}_Cij_ver"] - C0)[..., off].mean(-1))),
            "median_abs_mx_err_raw": float(np.median(np.abs(nz[f"p{p}_mx"] - x0))),
            "median_abs_mx_err_verified": float(np.median(np.abs(nz[f"p{p}_mx_ver"] - x0))),
            "median_accept": float(np.median(nz[f"p{p}_accept"])), "min_accept": float(np.min(nz[f"p{p}_accept"])),
            "clean_circuit_parity_min": float(np.min(nz["p0.0_parity"])),
            "note": "HVA preserves P exactly, so clean-circuit leakage is zero; all rejected weight is noise-induced"}
    # ---------------- first-order response vs measured shifts
    ph = pa.boundaries(K, H, *pa.orders(S0, N), T)["phase"]
    rows = []
    for i, kv in enumerate(K):
        if ph[i] < 0:
            continue
        sel = 0 if ph[i] == 0 else N // 4
        norm = (N - 1) if ph[i] == 0 else (N / 2 - 1)
        y0 = (S0[i, :, sel] - 1) / norm
        yp = (sq_from_cij(ex["plus_dp_Cij"][i])[:, sel] - 1) / norm
        ym = (sq_from_cij(ex["minus_dp_Cij"][i])[:, sel] - 1) / norm
        dOdp = (yp - ym) / (2 * 1e-4)
        hc = pa._crossing(H, y0, T)
        if not np.isfinite(hc):
            continue
        j = int(np.searchsorted(H, hc)) - 1
        slope = (y0[j + 1] - y0[j]) / (H[j + 1] - H[j])
        w = (hc - H[j]) / (H[j + 1] - H[j])
        dOdp_c = (1 - w) * dOdp[j] + w * dOdp[j + 1]
        r = {"kappa": float(kv), "h_T_p0": hc, "dO_dp_at_crossing": dOdp_c, "dO_dh_at_crossing": slope}
        for p in (0.01, 0.05):
            pred = -p * dOdp_c / slope
            yp_meas = (sq_from_cij(nz[f"p{p}_Cij"][i])[:, sel] - 1) / norm
            meas = pa._crossing(H, yp_meas, T) - hc
            r[f"dh_pred_first_order_p{p}"] = pred; r[f"dh_measured_p{p}"] = meas
            ymit = mitig[f"p{p}_lin_R3"]["OF" if ph[i] == 0 else "OA"][i]
            r[f"dh_after_ZNE_R3_p{p}"] = pa._crossing(H, ymit, T) - hc
            yver = (nz[f"p{p}_Sq_ver"][i, :, sel] - 1) / norm
            r[f"dh_parity_verified_p{p}"] = pa._crossing(H, yver, T) - hc
        rows.append(r)
    with open(AN / "first_order_vs_measured_N8_L4.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.4f}" for k, v in r.items()})
    pa.save_json(AN / "mitigation_summary_N8_L4.json", report)
    # figures: mitigated diagrams
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    KK, HH = np.meshgrid(K, H, indexing="ij")
    for ax, (key, title) in zip(axs, [("raw0.05", "raw p=0.05"), ("p0.05_lin_R3", "ZNE (linear p, λ=1,2,3) p=0.05"),
                                        ("ver0.05", "parity-verified p=0.05"), ("p0.01_lin_R3", "ZNE (linear p, λ=1,2,3) p=0.01")]):
        if key == "raw0.05":
            lab = pa.classify(*pa.orders(nz["p0.05_Sq"], N), nz["p0.05_mx"], T, Tx)
        elif key == "ver0.05":
            lab = pa.classify(*pa.orders(nz["p0.05_Sq_ver"], N), nz["p0.05_mx_ver"], T, Tx)
        else:
            lab = mitig[key]["lab"]
        fs.phase_map(ax, K, H, lab, None, title)
        fs.overlays(ax, legend=False, pe=False, lw=1.0)
    fs.phase_legend(fig, ncol=4); fig.subplots_adjust(bottom=0.22, wspace=0.08)
    fig.savefig(FIG / "fig_mitigation_diagrams.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, p in zip(axs, (0.01, 0.05)):
        k = np.array([r["kappa"] for r in rows])
        ax.axhline(0, color="#999", lw=0.7)
        ax.plot(k, [r[f"dh_measured_p{p}"] for r in rows], "o-", color="#eb6834", ms=3, label="measured δh (raw)")
        ax.plot(k, [r[f"dh_pred_first_order_p{p}"] for r in rows], "s--", color="#2a78d6", ms=3, label="first-order prediction −p∂_pO/∂_hO")
        ax.plot(k, [r[f"dh_after_ZNE_R3_p{p}"] for r in rows], "^-", color="#1baf7a", ms=3, label="after ZNE (R3)")
        ax.plot(k, [r[f"dh_parity_verified_p{p}"] for r in rows], "v:", color="#4a3aa7", ms=3, label="parity verified")
        ax.set_title(f"threshold-crossing shift vs clean circuit, p={p}"); ax.set_xlabel("κ"); ax.set_ylabel("δh")
    axs[0].legend(fontsize=9.4, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig_shift_prediction.png"); plt.close(fig)
    print(json.dumps(report, indent=1))
    for r in rows:
        print({k: round(v, 3) for k, v in r.items()})


if __name__ == "__main__":
    main()
