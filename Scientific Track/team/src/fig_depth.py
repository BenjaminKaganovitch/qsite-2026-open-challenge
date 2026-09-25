"""Depth trade-off: HVA L=1..4 (clean-optimised, fixed parameters) on kappa cuts 0.2, 0.3, 0.7, 0.8.
Total error vs exact ground state = preparation error (grows as L decreases) + noise error (grows with L)."""
import json
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt
import phase_analysis as pa
import scan_io as sio

FIG = sio.RESULTS / "figures"; AN = sio.RESULTS / "analysis"; N = 8
ROWS = [6, 9, 21, 24]; PS = (0.0, 0.01, 0.05)


def load(L, r):
    src = "noisy_hva_N8_L4_31x31" if L == 4 else f"noisy_np__vqe_hva_N8_L{L}_31x31"
    return np.load(sio.row_path(src, r))


def main():
    out = {}
    fig, axs = plt.subplots(2, 4, figsize=(14, 5.8))
    cols = {1: "#e87ba4", 2: "#eda100", 3: "#1baf7a", 4: "#2a78d6"}
    for c, r in enumerate(ROWS):
        ed = np.load(sio.row_path("ed_N8_pbc_31x31", r)); H = ed["h"]; kv = float(ed["kappa"])
        OFe, OAe = pa.orders(ed["Sq"], N); ph = 0 if OFe[0] >= OAe[0] else 1
        ye = (OFe, OAe)[ph]
        axs[0, c].plot(H, ye, color="k", lw=1.4, label="ED")
        for L in (1, 2, 3, 4):
            z = load(L, r)
            for p, ls in zip(PS, ("-", "--", ":")):
                y = pa.orders(z[f"p{p}_Sq"], N)[ph]
                if p in (0.0, 0.01):
                    axs[0, c].plot(H, y, color=cols[L], ls=ls, lw=1.1, label=f"L={L} p={p}" if c == 0 else None)
                err = float(np.mean(np.abs(y - ye)))
                hD = pa._deriv_peak(H, y)
                out[f"kappa{kv:.3f}_L{L}_p{p}"] = {"mean_abs_order_err_vs_ED": err, "h_D": hD, "h_D_ED": pa._deriv_peak(H, ye),
                                                   "cnots": 4 * N * L}
        axs[0, c].set_title(f"κ={kv:.2f}: {'O_F' if ph == 0 else 'O_A'} (solid p=0, dashed p=0.01)"); axs[0, c].set_xlabel("h")
        for p, mk in zip(PS, ("o", "s", "^")):
            e = [out[f"kappa{kv:.3f}_L{L}_p{p}"]["mean_abs_order_err_vs_ED"] for L in (1, 2, 3, 4)]
            axs[1, c].plot([1, 2, 3, 4], e, marker=mk, lw=1.2, label=f"p={p}", color=["#0b0b0b", "#eb6834", "#1baf7a"][PS.index(p)])
        axs[1, c].set_xlabel("HVA layers L (32 CNOTs per layer)"); axs[1, c].set_ylabel("mean |O_circ − O_ED| over h")
        axs[1, c].set_xticks([1, 2, 3, 4])
    axs[0, 0].legend(fontsize=6, frameon=False, ncol=2); axs[1, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Depth trade-off (fixed clean-optimised parameters): preparation error falls with L, noise error rises", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_depth_tradeoff.png"); plt.close(fig)
    best = {}
    for r in ROWS:
        kv = float(np.load(sio.row_path("ed_N8_pbc_31x31", r))["kappa"])
        for p in PS:
            e = {L: out[f"kappa{kv:.3f}_L{L}_p{p}"]["mean_abs_order_err_vs_ED"] for L in (1, 2, 3, 4)}
            best[f"kappa{kv:.3f}_p{p}"] = {"best_L": min(e, key=e.get), "errors": e}
    pa.save_json(AN / "depth_tradeoff.json", {"per_point": out, "best_depth": best})
    print(json.dumps(best, indent=0))


if __name__ == "__main__":
    main()
