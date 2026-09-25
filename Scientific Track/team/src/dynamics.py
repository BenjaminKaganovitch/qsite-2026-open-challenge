"""Phase B extension: quench dynamics as corroboration. Start in |+>^N (the h->infinity ground state),
evolve under H(kappa,h) with first-order Trotter steps U = e^{+i dt h X} e^{-i dt kappa ZZnnn} e^{+i dt ZZnn}
compiled exactly as one HVA layer (32 CNOTs per step at N=8, target depolarisation after each CNOT).
Loschmidt rate lambda(t) = -(1/N) ln <+|rho(t)|+>. Error budget separated into
(exact - Trotter) and (Trotter - noisy Trotter). Writes results/analysis/dynamics.json + fig_dynamics.png."""
from scan_io import set_single_thread
set_single_thread()
import json
import numpy as np
import annni_core as ac
import hva
import phase_analysis as pa
import scan_io as sio

N, DT, STEPS = 8, 0.2, 10
CUTS = {0.2: [0.2, 0.4, 0.6, 0.8, 1.2], 0.8: [0.2, 0.4, 0.6, 0.8, 1.2]}
PS = (0.0, 0.01)


def main():
    plus = np.full(2**N, 2 ** (-N / 2))
    out = {}
    for kv, hl in CUTS.items():
        for hv in hl:
            H = ac.hamiltonian_sparse(N, kv, hv).toarray()
            E, V = np.linalg.eigh(H)
            c0 = V.T @ plus
            ts = DT * np.arange(STEPS + 1)
            exact = np.array([abs(np.sum(np.abs(c0) ** 2 * np.exp(-1j * E * t))) ** 2 for t in ts])
            rec = {"t": ts.tolist(), "exact_echo": exact.tolist()}
            g = np.array([[-DT, kv * DT, -hv * DT]])
            for p in PS:
                ech = [1.0]
                for k in range(1, STEPS + 1):
                    rho = hva.gate_density_matrix(np.repeat(g, k, 0), N, k, p)
                    ech.append(float(np.real(plus @ rho @ plus)))
                rec[f"trotter_echo_p{p}"] = ech
            lam = lambda e: (-np.log(np.clip(np.asarray(e), 1e-300, None)) / N).tolist()
            rec["rate_exact"] = lam(exact); rec["rate_trotter"] = lam(rec["trotter_echo_p0.0"]); rec["rate_noisy_p0.01"] = lam(rec["trotter_echo_p0.01"])
            rec["max_trotter_err_rate"] = float(np.max(np.abs(np.array(rec["rate_trotter"]) - rec["rate_exact"])))
            rec["max_channel_err_rate_p0.01"] = float(np.max(np.abs(np.array(rec["rate_noisy_p0.01"]) - rec["rate_trotter"])))
            rec["t_first_peak_exact"] = float(ts[1:-1][np.argmax(np.diff(np.sign(np.diff(rec["rate_exact"]))) < 0)]) if np.any(np.diff(np.sign(np.diff(rec["rate_exact"]))) < 0) else None
            rec["max_rate_exact"] = float(np.max(rec["rate_exact"]))
            out[f"kappa{kv}_h{hv}"] = rec
            print(kv, hv, "max rate exact %.3f trot-err %.3f chan-err %.3f" % (rec["max_rate_exact"], rec["max_trotter_err_rate"], rec["max_channel_err_rate_p0.01"]), flush=True)
    pa.save_json(sio.RESULTS / "analysis" / "dynamics.json", out)
    import figstyle as fs
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, kv in zip(axs, CUTS):
        for i, hv in enumerate(CUTS[kv]):
            r = out[f"kappa{kv}_h{hv}"]; c = plt.cm.viridis(i / (len(CUTS[kv]) - 1))
            ax.plot(r["t"], r["rate_exact"], color=c, lw=1.5, label=f"h={hv}")
            ax.plot(r["t"], r["rate_trotter"], color=c, lw=0, marker="o", ms=3)
            ax.plot(r["t"], r["rate_noisy_p0.01"], color=c, lw=0.9, ls="--")
        ax.set_title(f"κ={kv}: quench from |+⟩, Loschmidt rate λ(t)\nline exact, dots Trotter dt={DT}, dashed Trotter + p=0.01"); ax.set_xlabel("t")
    axs[0].set_ylabel("λ(t) = −ln|⟨+|ψ(t)⟩|²/N"); axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(sio.RESULTS / "figures" / "fig_dynamics.png"); plt.close(fig)


if __name__ == "__main__":
    main()
