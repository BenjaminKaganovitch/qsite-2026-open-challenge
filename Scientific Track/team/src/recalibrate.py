"""Re-apply the anchor-region calibration rule (phase_analysis.calibrate) separately at each noise level,
using only the noisy circuit data (no ED input), and compare the resulting labels with the clean ED labels.
Writes results/analysis/recalibrated_thresholds.json."""
import json
import numpy as np
import phase_analysis as pa
import scan_io as sio

N = 8


def main():
    AN = sio.RESULTS / "analysis"
    cal_ed = json.loads((AN / "calibration_ed_N8.json").read_text())
    _, ed, _ = pa.load_scan("ed_N8_pbc_31x31", 31)
    _, nz, _ = pa.load_scan("noisy_hva_N8_L4_31x31", 31)
    K, H = ed["kappa"], ed["h"]
    KK, HH = np.meshgrid(K, H, indexing="ij")
    OFe, OAe = pa.orders(ed["Sq"], N)
    lab_ed = pa.classify(OFe, OAe, ed["mx"], cal_ed["T_order"], cal_ed["T_mx"])
    out = {"rule": "anchor medians (ferro kappa<=0.2&h<=0.2; antiphase kappa>=0.8&h<=0.1; para h>=1.8) applied to the circuit data at each p",
           "ED": {"T_order": cal_ed["T_order"], "T_mx": cal_ed["T_mx"], "counts": [int((lab_ed == i).sum()) for i in range(4)]}}
    for p in (0.0, 0.01, 0.05):
        OF, OA = pa.orders(nz[f"p{p}_Sq"], N); mx = nz[f"p{p}_mx"]
        c = pa.calibrate(OF, OA, mx, KK, HH)
        lab = pa.classify(OF, OA, mx, c["T_order"], c["T_mx"])
        out[f"p{p}"] = {"T_order": c["T_order"], "T_mx": c["T_mx"], "anchor_medians": c["anchor_medians"],
                        "counts": [int((lab == i).sum()) for i in range(4)],
                        "n_ferro": int((lab == 0).sum()), "agreement_with_ED": float(np.mean(lab == lab_ed)),
                        "per_phase_kept": {n: float(np.mean(lab[lab_ed == i] == i)) for i, n in enumerate(["ferro", "antiphase", "paramagnet"])}}
    pa.save_json(AN / "recalibrated_thresholds.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
