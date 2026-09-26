"""Build notebooks/ANNNI_phase_diagram_under_noise.ipynb (run: python src/make_notebook.py)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# ANNNI phase diagram under CNOT-target depolarizing noise
**Q-SITE Hacks 2026, Scientific Open Challenge.** Team: Benjamin, Dimitri, Yan, Frederick.

$$H=-\sum_i Z_iZ_{i+1}+\kappa\sum_i Z_iZ_{i+2}-h\sum_i X_i$$
Here $Z$ and $X$ are Pauli operators, $J_1=1$, and the ring is **periodic** (N nearest-neighbour and N next-nearest-neighbour bonds).

**Diagnostics.** $C(r)=\frac1N\sum_i\langle Z_iZ_{i+r}\rangle$, $S(q)=\frac1N\sum_{ij}e^{iq(i-j)}\langle Z_iZ_j\rangle$ (the self-terms $C_{ii}=1$ are kept) and $m_x=\frac1N\sum_i\langle X_i\rangle$. The order parameters subtract the uncorrelated floor: $O_F=(S(0)-1)/(N-1)$ and $O_A=(S(\pi/2)-1)/(N/2-1)$.

**Noise model.** `qml.DepolarizingChannel(p)` acts on the **target** qubit immediately after **every** CNOT, simulated on `default.mixed`. It is a single-qubit channel, not a two-qubit one.

**Noise protocol.** Circuit parameters are optimised once at p=0. The same circuits are then evaluated at p = 0, 0.01 and 0.05 with identical thresholds (fixed-parameter degradation).

**Overlays.** The Ising curve is a second-order estimate; the PT (lower) and KT (upper) curves are empirical 1.05 fits. They are thermodynamic references, not finite-ring labels. Citations such as [Sxx] refer to `team/literature/`.

The notebook regenerates every figure from cached arrays in `team/results/`. Set `RECOMPUTE=True` to rerun the scan CLIs, which is slow; the laptop and Mac commands are in `team/README.md`.""")
code(r"""import os, sys, json, csv, subprocess
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]: os.environ[v]="1"
from pathlib import Path
TEAM = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = TEAM / "src"; RES = TEAM / "results"; FIG = RES / "figures"; AN = RES / "analysis"
sys.path.insert(0, str(SRC))
from IPython.display import Image, display
import numpy as np
import annni_core as ac, scan_io as sio, phase_analysis as pa
RECOMPUTE = False      # True: rerun all scans (hours on one core; use the sharded CLIs instead)
print(json.dumps(ac.software_versions(), indent=1))
def show(name, width=900): display(Image(filename=str(FIG / name), width=width))
def js(name): return json.loads((AN / name).read_text())""")
md("## 1. Validation\nThe saved validation reports are shown below. Rerun them with `python src/validate_core.py` and `python src/validate_circuits.py`; each exits non-zero on failure.")
code(r"""core = json.loads((RES/"validation"/"core_validation.json").read_text())
circ = json.loads((RES/"validation"/"circuit_validation.json").read_text())
for k in ["H_vs_starter_and_pennylane_maxabs","bond_counts_N8_periodic","eigsh_vs_dense_maxdE","eigsh_max_residual","eigsh_max_orth_err",
          "ferro_S0","cat_S0","antiphase_Spi2","plus_mx","mixed_mx","hI_0","starter_hI_0_bug"]: print(f"{k:38s} {core[k]}")
for k in ["L4_state_vs_default.qubit","L4_adjoint_vs_fd","L4_p0.05_rho_vs_default.mixed","L4_p0_mixed_vs_pure","L4_cnots","L4_channels","L4_placement_ok","HVA_L4_cnots","HVA_L4_channels","HVA_L4_p","HVA_L4_placement_ok",
          "terminal_p0.05_Sq_formula_err"]: print(f"{k:38s} {circ[k]}")
import hva, circuits as cc
g = np.random.default_rng(3).uniform(-1,1,(4,3))
rho_np = hva.gate_density_matrix(g, 8, 4, 0.05); rho_pl = np.array(hva.pennylane_qnode(8, 4, 0.05)(g))
print("HVA N=8 L=4 p=0.05: NumPy engine vs default.mixed max|Δρ| =", np.abs(rho_np-rho_pl).max(), "; CNOTs =", hva.n_cnots(8,4))""")
md("## 2. Scans (sharded CLIs) and merge validation\nEvery scan stores one row file per κ row, and each row is a complete, warm-started sweep in h. The merge step checks that no row is missing or duplicated and that every row carries the same metadata. The discontinuity check flags any anomalous jump between neighbouring rows, which is also where shard boundaries fall.")
code(r"""if RECOMPUTE:
    py = sys.executable
    for cmd in [["ed_scan.py","--n","8"],["ed_scan.py","--n","10"],["ed_scan.py","--n","12"],
                ["ed_scan.py","--n","8","--nk","21","--nh","401","--hmin","0.005"],["ed_scan.py","--n","10","--nk","21","--nh","401","--hmin","0.005"],
                ["ed_scan.py","--n","12","--nk","21","--nh","401","--hmin","0.005"],["vqe_scan.py"],["noisy_eval.py","--mode","pennylane"],
                ["noisy_eval.py","--mode","extras"]]:
        subprocess.run([py, str(SRC/cmd[0])]+cmd[1:], check=True)
for scan, nr in [("ed_N8_pbc_31x31",31),("ed_N10_pbc_31x31",31),("ed_N12_pbc_31x31",31),("ed_N8_pbc_21x401",21),("ed_N12_pbc_21x401",21),
                 ("vqe_hva_N8_L4_31x31",31),("noisy_hva_N8_L4_31x31",31),("noisy_extras_hva_N8_L4_31x31",31)]:
    metas, d, info = sio.merge_rows(scan, nr)
    key = "Sq" if "Sq" in d else next(k for k in sorted(d) if k.endswith("_Sq"))
    rep = sio.boundary_discontinuity_report(d[key][..., 0], key+"[q=0]")
    print(f"{scan:32s} rows={len(metas)} version-sets={info['n_version_sets']} flagged-row-pairs={rep['flagged_row_pairs']} max/median jump={rep['max_row_jump']:.3f}/{rep['median_row_jump']:.3f}")""")
md("The flagged row pairs all fall at κ ∈ [0.37, 0.5], where the low-field ferro→antiphase level crossing makes S(0) jump. The same pairs are flagged in the exact-diagonalisation scans, and rows were assigned to shards in an interleaved pattern, so the jumps are physical rather than shard artefacts. The noisy and extras scans show no flags away from κ≈0.5.")
md("""## 3. Clean exact-diagonalisation reference (N=8, 31×31)
Sparse `eigsh` is run with k=6, warm-started along h, plus an independent cold start at every point. The maximum cold−warm energy difference is below 1e-13. When the ground state is degenerate within 1e-8·|E0|, observables are averaged equally over the degenerate subspace. Fidelity is computed between ground subspaces.

The thresholds are calibrated on **anchor regions that follow from the limiting cases, not from the overlay curves**: ferro at κ≤0.2 and h≤0.2, antiphase at κ≥0.8 and h≤0.1, and paramagnet at h≥1.8. Each threshold is the midpoint between the anchor medians.""")
code(r"""import fig_clean; fig_clean.main()
show("fig_ed_phase_N8.png", 520); show("fig_ed_observables_N8.png"); show("fig_ed_threshold_sensitivity_N8.png", 520)""")
md("""The ferro–para boundary follows the second-order Ising estimate closely, with finite-size deviations of 0.02 or less; the table in §7 gives the numbers. On the antiphase side the N=8 order-threshold boundary lies **above** the KT fit. Period-4 correlations persist over the whole 8-site ring even where the thermodynamic system would be floating or paramagnetic, so this is a finite-ring effect (§7 shows it drifting down with N).

An intermediate q* at high field (q*=π/4 at N=8) lies beyond the Peschel–Emery disorder line inside the gapped paramagnet [S10], so it is **not** labelled floating.""")
md("""## 4. Circuit preparation: Hamiltonian-variational ansatz (HVA) [S30]
The ansatz starts from $|+\\rangle^{N}$ and applies L layers of $e^{-ib\\sum X}e^{-ic\\sum ZZ_{nnn}}e^{-ia\\sum ZZ_{nn}}$, giving 3L parameters. Each ZZ bond compiles to CNOT–RZ–CNOT, so there are 4NL = 128 CNOTs at N=8, L=4.

**Why this ansatz:** a hardware-efficient RY/CNOT ring stalled at 1–3 % energy error near the boundaries even at L=6 (see `src/vqe.py`, benchmark in the report). The HVA preserves translation and parity symmetry, so it can represent the symmetric finite-ring ground state.

**Optimisation:** L-BFGS-B with an exact adjoint gradient. Each κ row is swept forward and backward in h with warm starts, the lower-energy branch is kept, and a repair pass follows. Four random starts are used at h=0 and two at h=2.""")
code(r"""import fig_noise; fig_noise.main()
print(json.dumps(js("prep_quality_N8_L4.json"), indent=1)); show("fig_circuit_vs_ed.png")""")
md("## 5. Required phase diagrams (fixed parameters, p = 0, 0.01, 0.05)\nPanel (a) uses the fixed clean-ED thresholds. Panel (b) uses a threshold-free relative-dominance label, argmax(O_F, O_A, m_x²). Under uniform *terminal* depolarisation all three quantities contract by the same factor a^{2m} [S28], so label (b) is exactly noise-invariant in that benchmark; any change it shows reveals circuit-specific (interleaved) noise action.")
code(r"""for p in (0.0, 0.01, 0.05): show(f"phase_diagram_p{p}.png", 820)
show("fig_phase_diagrams_all.png")
s = js("noise_summary_N8_L4.json"); print(json.dumps({k: v for k, v in s.items() if "kept" in k or "contraction" in k or "purity" in k or "accept" in k}, indent=1))""")
md("## 6. Quantitative noise analysis\nThe shift is δh(κ,p) = h_b(same circuit, p) − h_b(same circuit, 0), and preparation bias is reported separately as h_b(circuit, 0) − h_b(ED). The estimators are h_T (crossing of the fixed threshold), h_D (maximum of −dO/dh) and h_R (the relative-dominance boundary).")
code(r"""rows = list(csv.DictReader(open(AN/"boundary_shifts_circuit_N8_L4.csv")))
cols = ["kappa","order","h_T_ED","prep_bias_h_T","h_T_p0.0","h_T_p0.01","h_D_p0.0","dh_D_p0.01","dh_D_p0.05","h_R_p0.0","dh_R_p0.01","dh_R_p0.05"]
print(" ".join(f"{c:>11s}" for c in cols))
for r in rows[::3]: print(" ".join(f"{r[c]:>11s}" for c in cols))
print(json.dumps(js("seed_uncertainty_N8_L4.json")["median_abs_seed_difference"], indent=1))
show("fig_noise_cuts.png"); show("fig_contraction.png", 700)""")
md("### Mechanism: exact first-order channel-insertion responses [S28]\nd⟨O⟩/dp at p=0 equals the sum over all 128 channel locations of Tr{O_g L_t(ρ_g)}. Every location is evaluated separately and the sum is resolved by layer. For comparison, the terminal-equivalent value with 16 target hits per site would be d ln O/dp = −8·16/3 ≈ −43 for pair correlators and −21 for m_x.")
code(r"""ins = js("insertion_response_N8_L4.json")
for name, v in ins.items():
    print(f"{name:34s} O_F,O_A,m_x={np.round(v['O_F,O_A,m_x (p=0)'],3)}  dlnO/dp={np.round(v['relative dlnO/dp'],1)}  by layer (dominant O)={ {k: np.round(x,1).tolist() for k,x in v['by_layer'].items()} }")""")
md("## 7. Finite size (clean ED, N = 8, 10, 12) and floating-phase evidence")
code(r"""import fig_finite_size; fig_finite_size.main()
show("fig_finite_size_maps.png"); show("fig_finite_size_boundaries.png", 560); show("fig_floating_cuts.png")""")
md("""## 8. Error mitigation (fixed circuit parameters; p=0 truth held out of every fit)
**ZNE:** the noise is scaled linearly (p → λp, λ = 1, 2, 3), with a composed-channel path also tested at p=0.05. Richardson extrapolation is applied to each correlator $C_{ij}$ and each $\\langle X_i\\rangle$, and $S(q)$ is rebuilt afterwards; labels are never extrapolated.

**Exponential ZNE:** a two-point exponential extrapolation $C_0=C_1^2/C_2$ (λ = 1, 2; linear $2C_1-C_2$ where the two values differ in sign) is also applied to every $C_{ij}$ and to $m_x$. Its variance amplification is $4r^2+r^4$ with $r=C(p)/C(2p)$, reported below.

**Parity verification:** with $P=\\prod X_i$ and target sector s=+1, the verified value is $\\langle O\\rangle_+=(\\langle O\\rangle+\\langle OP\\rangle)/(1+\\langle P\\rangle)$ [S21].""")
code(r"""import fig_mitigation; fig_mitigation.main()
show("fig_mitigation_diagrams.png"); show("fig_shift_prediction.png", 760)
rc = js("recalibrated_thresholds.json")
for k in ("p0.0", "p0.01", "p0.05"):
    v = rc[k]; print(f"anchor rule re-applied at {k}: T={v['T_order']:.3f}, T_x={v['T_mx']:.3f}, ferro points={v['n_ferro']}, agreement with ED={v['agreement_with_ED']:.3f}")""")
md("## 9. Extensions: depth trade-off, unsupervised classifier, noise-aware re-optimisation, dynamics\nThese extensions use exactly the same noise model. The **depth** study reruns the fixed-parameter protocol for L = 1–4 on four κ cuts. **k-means** (k=3) is applied to $(S(q)/N, m_x)$ without labels or curves, and the clusters are named from their centroids afterwards. **Re-optimisation** is a separate protocol that minimises $\\mathrm{Tr}(H\\rho_p)$ at each p, using an exact gradient through the channels.")
code(r"""import fig_depth, fig_cluster, fig_reopt
fig_depth.main(); show("fig_depth_tradeoff.png")
fig_cluster.main(); show("fig_kmeans.png")
fig_reopt.main()
if (FIG/"fig_reopt.png").exists(): show("fig_reopt.png")""")
md("### Quench dynamics (corroboration only)\nThe ring starts in $|+\\rangle^N$ and evolves under first-order Trotter steps; each step is compiled exactly as one HVA layer (32 CNOTs, target noise after every CNOT). We plot the Loschmidt rate $\\lambda(t)=-\\ln|\\langle+|\\psi(t)\\rangle|^2/N$ and keep the Trotter error (exact vs Trotter) separate from the channel error (Trotter vs noisy Trotter). Rerun with `python src/dynamics.py` (about 1 min).")
code(r"""dyn = js("dynamics.json")
for k, v in dyn.items(): print(f"{k:16s} max λ exact={v['max_rate_exact']:.3f}  Trotter err={v['max_trotter_err_rate']:.3f}  channel err (p=0.01)={v['max_channel_err_rate_p0.01']:.3f}")
show("fig_dynamics.png", 800)""")
md("## 10. Conclusions and limitations\nThe write-up is `writeup/report.pdf` (source `writeup/report.tex`). The main caveats: N=8–12 rings; overlays are approximate thermodynamic curves; the noisy maps describe prepared-state diagnostics, while the Hamiltonian spectrum itself is unchanged; the full-grid maps use the fixed-parameter protocol, and noise-aware re-optimisation was run on four κ cuts (§9); at p=0.05 linear Richardson ZNE fails, while two-point exponential ZNE recovers the map only at the cost of a very large variance amplification (see `results/analysis/zne_exponential.json`).")
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent.parent / "notebooks" / "ANNNI_phase_diagram_under_noise.ipynb"
nbf.write(nb, out); print("wrote", out)
