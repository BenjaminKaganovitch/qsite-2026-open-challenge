"""Optimizer-seed uncertainty: independent clean HVA optimisations (seed 777) on selected kappa rows,
re-evaluated with the validated NumPy density-matrix engine at p=0, 0.01, 0.05, compared with the
main run (seed 2026). Output: results/analysis/seed_uncertainty_N8_L4.json"""
import json
import numpy as np
import phase_analysis as pa
import scan_io as sio

AN = sio.RESULTS / "analysis"; N = 8


def main():
    cal = json.loads((AN / "calibration_ed_N8.json").read_text()); T = cal["T_order"]
    alt = "noisy_np__vqe_hva_N8_L4_31x31_seed777"
    rows = sorted(int(f.stem.split("_")[1]) for f in (sio.RESULTS / alt).glob("row_*.npz"))
    out = {"rows": rows, "per_row": {}}
    diffs = {k: [] for k in ["dE_rel", "h_T_p0.0", "h_D_p0.0", "h_D_p0.01", "h_D_p0.05", "h_T_p0.01", "dh_D_p0.01"]}
    for r in rows:
        with np.load(sio.row_path("noisy_hva_N8_L4_31x31", r)) as a, np.load(sio.row_path(alt, r)) as b, \
             np.load(sio.row_path("vqe_hva_N8_L4_31x31", r)) as va, np.load(sio.row_path("vqe_hva_N8_L4_31x31_seed777", r)) as vb:
            H = a["h"]; kv = float(a["kappa"])
            rec = {"kappa": kv, "max_rel_dE_between_seeds": float(np.max(np.abs(va["E_vqe"] - vb["E_vqe"]) / np.abs(va["E_ed"])))}
            OF0, OA0 = pa.orders(a["p0.0_Sq"], N)
            ph = 0 if OF0[0] >= OA0[0] else 1
            for p in (0.0, 0.01, 0.05):
                ya = pa.orders(a[f"p{p}_Sq"], N)[ph]; yb = pa.orders(b[f"p{p}_Sq"], N)[ph]
                for est, f in (("h_T", lambda y: pa._crossing(H, y, T)), ("h_D", lambda y: pa._deriv_peak(H, y))):
                    rec[f"{est}_p{p}"] = [f(ya), f(yb)]
            rec["dh_D_p0.01"] = [rec["h_D_p0.01"][0] - rec["h_D_p0.0"][0], rec["h_D_p0.01"][1] - rec["h_D_p0.0"][1]]
            out["per_row"][str(r)] = rec
            diffs["dE_rel"].append(rec["max_rel_dE_between_seeds"])
            for k in ["h_T_p0.0", "h_D_p0.0", "h_D_p0.01", "h_D_p0.05", "h_T_p0.01", "dh_D_p0.01"]:
                v = rec[k]; diffs[k].append(abs(v[0] - v[1]) if np.all(np.isfinite(v)) else np.nan)
    out["max_abs_seed_difference"] = {k: float(np.nanmax(v)) if np.any(np.isfinite(v)) else None for k, v in diffs.items()}
    out["median_abs_seed_difference"] = {k: float(np.nanmedian(v)) if np.any(np.isfinite(v)) else None for k, v in diffs.items()}
    pa.save_json(AN / "seed_uncertainty_N8_L4.json", out)
    print(json.dumps(out["max_abs_seed_difference"], indent=1)); print(json.dumps(out["median_abs_seed_difference"], indent=1))


if __name__ == "__main__":
    main()
