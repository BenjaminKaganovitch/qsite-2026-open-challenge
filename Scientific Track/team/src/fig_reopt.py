"""Noise-aware re-optimisation vs fixed-parameter degradation on kappa cuts (explicitly separate protocols)."""
import json
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt
import phase_analysis as pa
import scan_io as sio

FIG = sio.RESULTS / "figures"; AN = sio.RESULTS / "analysis"; N = 8
SCAN = "reopt_hva_N8_L4_31x31_s4"


def main():
    rows = sorted(int(f.stem.split("_")[1]) for f in (sio.RESULTS / SCAN).glob("row_*.npz"))
    if not rows:
        print("no re-optimisation rows yet"); return
    out = {}
    fig, axs = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 3.2), squeeze=False)
    for ax, r in zip(axs[0], rows):
        z = np.load(sio.row_path(SCAN, r)); fx = np.load(sio.row_path("noisy_hva_N8_L4_31x31", r)); ed = np.load(sio.row_path("ed_N8_pbc_31x31", r))
        H = z["h"]; kv = float(z["kappa"]); Hf = fx["h"]; idx = [int(np.argmin(abs(Hf - x))) for x in H]
        OFe, OAe = pa.orders(ed["Sq"], N); ph = 0 if OFe[0] >= OAe[0] else 1
        ax.plot(Hf, (OFe, OAe)[ph], color="k", lw=1.3, label="ED")
        rec = {"kappa": kv, "h": H.tolist()}
        for p, c in ((0.01, "#eb6834"), (0.05, "#1baf7a")):
            yf = pa.orders(fx[f"p{p}_Sq"], N)[ph]
            yr = pa.orders(z[f"p{p}_Sq"], N)[ph]
            ax.plot(Hf, yf, color=c, ls="--", lw=1, label=f"fixed p={p}")
            ax.plot(H, yr, color=c, marker="o", ms=3, lw=1.2, label=f"re-optimised p={p}")
            rec[f"p{p}"] = {"E_noisy_fixed": z[f"p{p}_E_noisy_fixed"].tolist(), "E_noisy_reopt": z[f"p{p}_E_noisy_reopt"].tolist(),
                            "E_clean_of_reopt_params": z[f"p{p}_E_clean_of_reopt_params"].tolist(),
                            "order_fixed": yf[idx].tolist(), "order_reopt": yr.tolist(),
                            "mean_order_gain": float(np.mean(yr - yf[idx])),
                            "mean_abs_err_vs_ED_fixed": float(np.mean(np.abs(yf[idx] - (OFe, OAe)[ph][idx]))),
                            "mean_abs_err_vs_ED_reopt": float(np.mean(np.abs(yr - (OFe, OAe)[ph][idx]))),
                            "mx_fixed": fx[f"p{p}_mx"][idx].tolist(), "mx_reopt": z[f"p{p}_mx"].tolist()}
        ax.set_title(f"κ={kv:.2f}: {'O_F' if ph == 0 else 'O_A'}"); ax.set_xlabel("h")
        out[str(r)] = rec
    axs[0, 0].legend(fontsize=8.7, frameon=False)
    fig.suptitle("Noise-aware re-optimisation (min Tr Hρ_p) vs fixed clean parameters, HVA L=4", fontsize=13.0)
    fig.tight_layout(); fig.savefig(FIG / "fig_reopt.png"); plt.close(fig)
    pa.save_json(AN / "reopt_vs_fixed.json", out)
    for r, v in out.items():
        print(r, v["kappa"], {p: (round(v[p]["mean_order_gain"], 3), round(v[p]["mean_abs_err_vs_ED_fixed"], 3), round(v[p]["mean_abs_err_vs_ED_reopt"], 3)) for p in ("p0.01", "p0.05")})


if __name__ == "__main__":
    main()
