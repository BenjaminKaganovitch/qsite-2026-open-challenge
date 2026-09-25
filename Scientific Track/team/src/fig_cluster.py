"""Alternative, unsupervised classifier: k-means on physical features [S(q)/N for q in 0..pi, m_x]
of clean ED N=8 states (no labels, no overlay curves used). Clusters are named afterwards from their
centroids; agreement with the rule-based labels is reported as *consistency*, not independent validation.
The same centroids are then applied to the noisy circuit data (p=0.01, 0.05)."""
import json
import numpy as np
from itertools import permutations
from scipy.cluster.vq import kmeans2
import figstyle as fs
import matplotlib.pyplot as plt
import phase_analysis as pa
import scan_io as sio

FIG = sio.RESULTS / "figures"; AN = sio.RESULTS / "analysis"; N = 8


def feats(Sq, mx):
    return np.concatenate([Sq[..., : N // 2 + 1] / N, mx[..., None]], axis=-1)


def main():
    cal = json.loads((AN / "calibration_ed_N8.json").read_text())
    _, ed, _ = pa.load_scan("ed_N8_pbc_31x31", 31)
    _, nz, _ = pa.load_scan("noisy_hva_N8_L4_31x31", 31)
    K, H = ed["kappa"], ed["h"]
    X = feats(ed["Sq"], ed["mx"]).reshape(-1, N // 2 + 2)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    cent, lab = kmeans2(Z, 3, seed=np.random.default_rng(2026), minit="++", iter=100)
    # name clusters from centroids (in original units): largest S(0) -> ferro, largest S(pi/2) -> antiphase, rest para
    C = cent * sd + mu
    names = {}
    names[int(np.argmax(C[:, 0]))] = 0
    names[int(np.argmax(C[:, 2]))] = 1
    for i in range(3):
        names.setdefault(i, 2)
    labk = np.vectorize(names.get)(lab).reshape(len(K), len(H))
    OF, OA = pa.orders(ed["Sq"], N)
    rule = pa.classify(OF, OA, ed["mx"], cal["T_order"], cal["T_mx"])
    agree = float(np.mean(labk == rule))
    res = {"k": 3, "features": "S(q)/N for q=0,pi/4,pi/2,3pi/4,pi and m_x (standardised)",
           "centroids_original_units": C.tolist(), "cluster_names": {str(k): pa.LABELS[v] for k, v in names.items()},
           "agreement_with_rule_labels_clean_ED": agree}
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    fs.phase_map(axs[0], K, H, labk, None, f"k-means (k=3), clean ED: {100*agree:.1f}% = rule labels")
    for ax, p in zip(axs[1:], (0.01, 0.05)):
        Xp = ((feats(nz[f"p{p}_Sq"], nz[f"p{p}_mx"]).reshape(-1, N // 2 + 2)) - mu) / sd
        d = ((Xp[:, None, :] - cent[None]) ** 2).sum(-1)
        lp = np.vectorize(names.get)(np.argmin(d, 1)).reshape(len(K), len(H))
        res[f"agreement_noisy_p{p}_vs_clean_kmeans"] = float(np.mean(lp == labk))
        fs.phase_map(ax, K, H, lp, None, f"clean centroids applied to circuit p={p}")
    for ax in axs:
        fs.overlays(ax, legend=False, pe=False, lw=1.0)
    fs.phase_legend(fig, ncol=4); fig.subplots_adjust(bottom=0.22, wspace=0.06)
    fig.savefig(FIG / "fig_kmeans.png"); plt.close(fig)
    pa.save_json(AN / "kmeans_classifier.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "centroids_original_units"}, indent=1))


if __name__ == "__main__":
    main()
