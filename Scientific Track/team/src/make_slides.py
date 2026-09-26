"""Build writeup/slides.pdf (16:9) from saved figures and analysis files."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import scan_io as sio

FIG = sio.RESULTS / "figures"; OUT = sio.TEAM / "writeup"
INK, INK2, ACC = "#0b0b0b", "#52514e", "#2a78d6"


def slide(pdf, title, bullets=(), images=(), subtitle=None, img_boxes=None, text_box=(0.05, 0.12, 0.40, 0.72), fs=13):
    fig = plt.figure(figsize=(13.33, 7.5)); fig.patch.set_facecolor("white")
    fig.text(0.05, 0.92, title, fontsize=24, weight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.05, 0.855, subtitle, fontsize=13, color=INK2, va="top")
    import textwrap
    x, y, w, h = text_box
    yy = y + h
    width_chars = int(w * 13.33 * 72 / (fs * 0.55))
    for b in bullets:
        indent = b.startswith("  ")
        lines = textwrap.wrap(b.strip(), width_chars - (3 if indent else 0))
        txt = ("– " if indent else "• ") + ("\n   ".join(lines))
        fig.text(x + (0.02 if indent else 0), yy, txt, fontsize=fs - (1 if indent else 0),
                 color=INK if not indent else INK2, va="top", linespacing=1.25)
        yy -= (0.058 if not indent else 0.05) + 0.034 * (len(lines) - 1) + 0.012
    for (path, box) in zip(images, img_boxes or []):
        ax = fig.add_axes(box); ax.imshow(mpimg.imread(FIG / path)); ax.axis("off")
    fig.text(0.95, 0.03, "https://github.com/BenjaminKaganovitch/qsite-2026-open-challenge", fontsize=8, color=INK2, ha="right")
    pdf.savefig(fig); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    with PdfPages(OUT / "slides.pdf") as pdf:
        slide(pdf, "ANNNI phase diagram under CNOT-target noise",
              ["Exact reference (sparse ED, N = 8, 10, 12) and a symmetry-preserving variational circuit (N = 8, 128 CNOTs)",
               "Same circuit evaluated at p = 0, 0.01, 0.05 with DepolarizingChannel(p) on every CNOT target (default.mixed)",
               "Headline: p = 0.01 erases the ferromagnet under fixed thresholds, yet derivative-defined boundaries barely move;",
               "  p = 0.05 leaves an essentially maximally mixed state; ZNE recovers the p = 0.01 map (98 % label agreement)",
               "Floating phase: compatible signatures at N = 12, not resolved"],
              subtitle="Benjamin · Dimitri · Yan · Frederick",
              images=["fig_phase_diagrams_all.png"], img_boxes=[[0.05, 0.03, 0.9, 0.45]], text_box=(0.05, 0.5, 0.9, 0.3), fs=14)
        slide(pdf, "Model, order parameters and reference curves",
              ["H = −Σ Z_iZ_{i+1} + κ Σ Z_iZ_{i+2} − h Σ X_i, Pauli ops, periodic ring",
               "Finite symmetric ground states have ⟨Z_i⟩ = 0 → use correlations",
               "S(q) = (1/N) Σ_ij e^{iq(i−j)} ⟨Z_iZ_j⟩ (C_ii = 1 kept)",
               "  O_F = (S(0)−1)/(N−1),  O_A = (S(π/2)−1)/(N/2−1),  m_x",
               "  checks: ferro S(0)=N, antiphase S(π/2)=N/2, |+⟩ and I/2^N both S=1",
               "Overlays are approximate thermodynamic curves:",
               "  Ising 2nd-order estimate (h_I(0)=1), lower PT fit, upper KT fit",
               "Thresholds calibrated on limiting-case anchors, not on the curves"],
              images=["fig_ed_observables_N8.png"], img_boxes=[[0.04, 0.04, 0.92, 0.28]], text_box=(0.05, 0.36, 0.9, 0.47), fs=13)
        slide(pdf, "Methods and validation",
              ["Sparse eigsh, k = 6, warm + independent cold start (ΔE < 1e-13)",
               "HVA: |+⟩ then 4 × [ZZ_nn, ZZ_nnn, X] rotations, 12 parameters",
               "  RY/CNOT ring stalled at 1–3 % energy error → switched",
               "  adjoint-gradient L-BFGS, forward + reverse warm sweeps",
               "Clean circuit vs ED: median relE 4e-5, median 1−F 5e-5, labels 98.9 %",
               "Noise: 128 target channels; tape check; NumPy DM engine = default.mixed to 3e-14",
               "Sharded, resumable CLIs; every figure regenerated from .npz"],
              images=["fig_circuit_vs_ed.png"], img_boxes=[[0.04, 0.04, 0.92, 0.3]], text_box=(0.05, 0.38, 0.9, 0.45), fs=13)
        slide(pdf, "Exact phase diagram and finite-size behaviour",
              ["Ferro–para boundary ≈ N-independent, tracks the Ising estimate",
               "  κ=0.2: χ_F peak 0.616 (N=8) → 0.629 (N=12); DMRG 0.639",
               "Antiphase boundary drifts down with N toward the KT fit",
               "  κ=0.75: 0.562 (N=8) → 0.432 (N=12); KT fit 0.423",
               "N = 10 cannot host period 4: commensurability anomaly",
               "Floating: N=12 entropy peak near KT line, q* locked at π/2 → unresolved"],
              images=["fig_finite_size_boundaries.png", "fig_floating_cuts.png"], img_boxes=[[0.52, 0.45, 0.30, 0.42], [0.5, 0.06, 0.48, 0.37]], text_box=(0.05, 0.12, 0.43, 0.72), fs=13)
        slide(pdf, "Effect of depolarising noise at p = 0.01 and 0.05",
              ["p=0.01: order contracts to 0.60 (median) of its clean value",
               "  ferro O_F(h→0) 1.00 → 0.37; antiphase O_A 1.00 → 0.59",
               "  fixed threshold: ferro 0 % kept, antiphase 46 %, para 89 %",
               "  surviving threshold crossings move to lower h (−0.04…−0.48)",
               "  derivative peaks: median |δh| 0.005 (28/31 rows ≤ 0.03)",
               "p=0.05: purity 0.0047 vs 0.0039 maximally mixed; all 'unpolarised'",
               "Noise does not change H: these are prepared-state diagnostics"],
              images=["fig_noise_cuts.png"], img_boxes=[[0.5, 0.18, 0.48, 0.62]], text_box=(0.05, 0.12, 0.43, 0.72), fs=13)
        slide(pdf, "Channel-insertion analysis of noise sensitivity",
              ["d⟨O⟩/dp = Σ_g Tr{O_g L_t(ρ_g)} over all 128 channel locations [S28]",
               "d ln O/dp: ferro −99, antiphase −69, m_x (para) −32…−48",
               "  terminal-noise equivalent (16 hits/site): −43 for pair correlators",
               "Mid-circuit errors are carried by later CNOT–RZ–CNOT gadgets",
               "Early layers contribute most (ferro: −27 layer 1 vs −17 layer 4)",
               "First-order −p∂_pO/∂_hO has the right sign; measured shifts 1.5–4× larger (nonlinear)",
               "Robustness ranking (ferro weakest) is circuit- and diagnostic-specific"],
              images=["fig_shift_prediction.png"], img_boxes=[[0.5, 0.25, 0.48, 0.45]], text_box=(0.05, 0.12, 0.43, 0.72), fs=13)
        slide(pdf, "Circuit depth, re-optimisation and clustering",
              ["Same experiment for HVA depth L = 1…4 (32 CNOTs per layer) on four κ cuts",
               "Clean error falls with L; noise error rises with L",
               "  best L at p=0.01: 3 (ferro), 2 (antiphase); at p=0.05: 1–2",
               "Antiphase is accurate already at L=2 (error 0.024); ferro needs L=4",
               "  the GHZ-like ferro state uses the full depth, so it carries the most channels",
               "Noise-aware re-optimisation (min Tr Hρ_p): energy drops 0.2–1.0, order moves ≤ 0.03",
               "  optimiser buys energy via m_x by leaving the ground state: energy-optimal ≠ diagnostic-optimal",
               "k-means on (S(q)/N, m_x), no labels: 99.1 % agreement with rule labels (consistency)"],
              images=["fig_depth_tradeoff.png"], img_boxes=[[0.12, 0.07, 0.76, 0.29]], text_box=(0.05, 0.41, 0.9, 0.45), fs=12)
        slide(pdf, "Error mitigation by extrapolation and parity checks",
              ["ZNE: Richardson on each C_ij and ⟨X_i⟩ at λp, λ = 1, 2, 3; p=0 held out",
               "  p=0.01: |ΔC_ij| 0.063 → 0.011, labels 70 % → 98 %, |δh| ≤ 0.09",
               "  cost: variance × 19;  p=0.05: fails (signal ~1 %)",
               "Parity verification P = ΠX_i: acceptance 71 % (p=0.01)",
               "  |ΔC_ij| 0.063 → 0.053: most errors stay in the even sector",
               "Mitigation cannot fix finite-N physics or ansatz bias"],
              images=["fig_mitigation_diagrams.png"], img_boxes=[[0.04, 0.04, 0.92, 0.33]], text_box=(0.05, 0.4, 0.9, 0.43), fs=13)
        slide(pdf, "Conclusions and limitations",
              ["Accurate clean diagram on 8–12-site rings; ferro side converged, antiphase side drifting to KT",
               "For a circuit good enough to represent these ground states (128 CNOTs):",
               "  p = 0.01 removes absolute order (ferro first) while boundary positions stay put",
               "  p = 0.05 leaves no usable signal; depth vs noise is the real trade-off",
               "ZNE turns p = 0.01 back into a correct diagram; parity checks help little",
               "Quench dynamics (Trotter from |+⟩): Loschmidt-rate peaks only for quenches deep into the ordered side (h ≤ 0.4); channel error at p=0.01 (≈0.28) exceeds the Trotter error (≤0.035)",
               "Limits: N ≤ 12, approximate overlays, κ≈0.5 masked, fixed-parameter protocol only, no shot noise",
               "All code, data and notebook: https://github.com/BenjaminKaganovitch/qsite-2026-open-challenge"],
              text_box=(0.05, 0.12, 0.9, 0.7), fs=16)
    print("wrote", OUT / "slides.pdf")


if __name__ == "__main__":
    main()
