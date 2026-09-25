# What the literature says about this ANNNI challenge

Literature-only research, 2026-09-25. **Literature base complete with two full-text access gaps:** see [LITERATURE_INDEX.md](LITERATURE_INDEX.md) for the source-by-source reading audit. The conclusions below distinguish verified source claims from our algebra and proposed experimental choices. No project code or simulation results were produced.

Use fixed citations `[Sxx]`, linked to the corresponding source note. Each note contains the write-up-ready bibliographic citation, source URL, access status, equations, parameters, limitations and rubric relevance. A source’s FULL status refers to the exact version/scope identified there, not all subsequent versions or every linked reference.

## 1. Which Hamiltonian and boundary conditions are actually used?

Our reference convention is Pauli operators, lattice spacing 1, J1=1:

\[
H(\kappa,h)=-\sum_iZ_iZ_{i+1}+\kappa\sum_iZ_iZ_{i+2}-h\sum_iX_i.
\]

| Source | Executable convention | Boundary conditions | Important difference |
|---|---|---|---|
| Starter | −j1 ZZ+j1κ NNN ZZ−hX | Periodic default; open optional | Normalized field is h/j1, not h if j1≠1 |
| PennyLane ANNNI demo | −XX+κ NNN XX−hZ | Open | Global Hadamard on every site converts XX/Z to ZZ/X |
| CERN exact builder | −XX+κ NNN XX−hZ | `ring=False` default | Set ring=True for comparable N=8–12 PBC |
| CERN DMRG wrapper | Same Pauli axes as demo | Finite/open | Replaces requested field below 0.1 by 0.1 |
| BCF DMRG papers | −J1 ZZ−J2 NNN ZZ−BX | Open/fixed-end protocols | κ=−J2/J1,h=B/J1 |

These are Pauli matrices with eigenvalues ±1. If using true spin operators S=σ/2, two-site coefficients must be multiplied by 4 and field coefficients by 2 to represent the same Hamiltonian. Merely swapping a library’s convention changes the relative field scale. [S02](S02.md), [S05](S05.md), [S06](S06.md), [S07](S07.md), [S10](S10.md), [S24](S24.md).

For N=8–12 the periodic starter has N nearest and N next-nearest terms; OBC has N−1 and N−2. Do not compare OBC and PBC vectors as if related solely by a basis rotation. The ANNNI demo’s displayed Hamiltonian contains sign discrepancies with its code; reproduce the competing-coupling code convention. Its states come from dense ED plus StatePrep, **not an implemented VQE**. The VQE ansatz belongs to the Monaco paper: RY layers and linear CNOT entanglers. [S02](S02.md), [S11](S11.md).

The CERN data interface is also not plug-and-play: exact-state pickles hold amplitudes, MPS pickles hold tensors, Brick pickles hold preparation unitaries; the axis order, class encoding and anomaly-loss target differ from the demo. No precomputed states or trained checkpoints are tracked in the current checkout. However, the linked Zenodo v1.0.0 archive contains saved VQE parameters and energies for open N=4,6,8,10,12 and periodic N=6,8,10 on 100×100 grids. Its old Hamiltonian is −XX−K NNN XX−LZ, so κ=−K and h=L. The extensionless pickle files include legacy Hamiltonian objects and circuit references; they are not ready-to-use raw states or verified current-version checkpoints. [S06](S06.md).

## 2. Where do the three boundary curves come from, and how accurate are they for N=8–12?

Write the Ising approximation as

\[
h_I(\kappa)\approx\frac{1-\kappa}{\kappa}\left[1-\sqrt{\frac{1-3\kappa+4\kappa^2}{1-\kappa}}\right],\quad0<\kappa<\tfrac12.
\]

Our algebraic rationalization, numerically better behaved at κ=0, is

\[
h_I(\kappa)=\frac{2(1-2\kappa)}{1+\sqrt{(1-3\kappa+4\kappa^2)/(1-\kappa)}}.
\]

It has hI(0)=1 and hI(1/2)=0. It is the smaller root of the **second-order gap estimate** h−κh²/[2(1−κ)]=1−2κ, not an exact general solution. BCF (2006) attributes the second-order estimate to Peschel–Emery. The original 1981 full text was inaccessible, so that provenance remains indirect; the downloaded BCFv1 also has printed/extracted sign inconsistencies against its own table and the later convention. We use the later consistent expression and flag this rather than silently claiming the original derivation was inspected. [S09](S09.md), [S10](S10.md), [S12](S12.md).

The **Peschel–Emery disorder line** is a different object:

\[
h_{PE}=\frac1{4\kappa}-\kappa,\quad0<\kappa<\tfrac12.
\]

It lies within the gapped paramagnet and marks a correlation-modulation crossover in the BCF study. An oscillating correlation is therefore not by itself evidence of a floating phase. [S10](S10.md).

Forκ>1/2, BCF (2007) Eq.(18) supplies empirical fits

\[
h_{\mathrm{lower}}\approx1.05(\kappa-.5),\qquad
h_{\mathrm{upper}}\approx1.05\sqrt{(\kappa-.5)(\kappa-.1)}.
\]

The coefficient 1.05 is fitted, not a universal exact constant. Cea identifies the lower antiphase–floating transition as **Pokrovsky–Talapov (PT)** and the upper floating–paramagnet transition as **KT/BKT**. BKT and KT are names for the same transition family; they should not be used to distinguish these two different boundaries. Plot labels should say “lower PT reference (organiser `bkt_transition`)” and “upper KT reference.” [S01](S01.md), [S12](S12.md), [S13](S13.md).

The high-frustration study includesκ from 0.5 to 5, but its fits remain numerical guides in the thermodynamic setting, not certified errors over arbitrary extrapolations. Atκ=.75 the lower fit gives.2625 versus numerical.2574(2); the upper gives about.4233 versus gap estimate.424(3) and entropy estimate.44(1). Different diagnostics and finite-size extrapolations matter. [S13](S13.md).

**There is no defensible universal numerical offset for periodic N=8–12 in these sources.** BCF (2006) TableI already shows thermodynamic Ising approximation bias: at κ=.2, .65336 versus.6393(1), a difference≈.0141; at κ=.3, .44183 versus.43669(4), difference≈.00514. These are not small-ring finite-size errors. The high-frustration work uses much larger chains and encounters extremely long KT correlation lengths. [S10](S10.md), [S13](S13.md).

Our finite-size consequences: N=8 and 12 accommodate period4; N=10 frustrates it. Allowed momenta are2πm/N, so Δq=π/4 at N=8 and π/6 at N=12. A 15-point h grid over[0,2] has spacing 1/7≈.143, already much coarser than many quoted reference differences. Treat observed crossings as finite-size pseudocritical estimates, refine selected cuts, and report grid/threshold/size drift separately from curve-approximation bias. Do not invent an expected percent agreement. These are arithmetic and finite-ring deductions, not published N=8–12 predictions. Conventional power-law finite-size fits also cannot be transferred blindly to KT transitions: the essential divergence gives logarithmically slow pseudocritical drift, with important corrections. [S23](S23.md).

## 3. Which diagnostics can separate the phases at these sizes?

Use the translation average C(r)=N⁻¹Σi⟨ZiZi+r⟩ and

\[
S(q)=\frac1N\sum_{ij}e^{iq(i-j)}\langle Z_iZ_j\rangle,\qquad
m^2(q)=S(q)/N.
\]

The handoff calls the second normalization “S(q).” Either convention works, but label it and retain Cii=1 consistently. Sandvik Eq.(68) uses the first normalization. [S23](S23.md), [S29](S29.md).

| Regime | Main evidence | Cross-checks and caveats |
|---|---|---|
| Ferro | Large S(0)/N; positive longer-distance ZZ | Symmetry makes ⟨Z⟩ vanish in finite exact states |
| Antiphase | Peak atπ/2 on a commensurate ring; negative C(2), period-four pattern | Use N divisible by 4; NN correlation alone can average to0 |
| Paramagnet | Weak longitudinal ordering plus large mx=N⁻¹Σ⟨X⟩ | Infinite-temperature noise also weakens ZZ but has mx≈0 |
| Floating candidate | Nonlocked modulation, extended correlations and consistent size-dependent low-energy/entropy evidence | A finite-q peak can also occur in a gapped modulated paramagnet |

Own limiting-state checks: all-up/all-down ferro has S(0)=N; a perfect period-four pattern on N divisible by 4 has S(π/2)=N/2; both |+⟩^N and I/2^N have S(q)=1, but mx=1 and0 respectively. Symmetry-restored cats retain suitable two-point order while their one-point magnetizations vanish. The starter’s `z_abs_mean` does not cure this, and its `antiphase_string_mean` is a product of expectations that can be misleading in both ordered phases. [S07](S07.md), [S28](S28.md).

Fidelity susceptibility supplements these observables. Define squared F=|⟨ψ(λ)|ψ(λ+δ)⟩|² and χ≈−ln(F)/δ². Gu’s amplitude convention is f=√F with χ≈−2ln(f)/δ². Consequently the handout’s −F″ is 2χ at δ=0. State the direction in(κ,h), step, and normalization per site if used. Near degeneracy use ground-subspace comparison or symmetry tracking; a single eigenvector can rotate arbitrarily. A small E1−E0 alone may be an ordered-phase tunnel splitting. [S15](S15.md), [S16](S16.md), [S13](S13.md).

For a pure periodic critical ground state, Sℓ=(c/3)ln[(N/πa)sin(πℓ/N)]+constant. OBC has c/6. Fitting c≈1 from N=8–12 is suggestive at best, especially with corrections and short distances. Noisy reduced entropy is not a clean entanglement diagnostic. [S17](S17.md).

**Evidence standard for floating:** require several independent indicators on a resolved parameter interval, test multiple commensurate sizes, examine correlation decay against both exponential and algebraic alternatives, and separate ordered degeneracies from genuinely low-energy critical excitations. N=8–12 offers too little distance range for a persuasive algebraic-decay fit. Cea’s own autoencoder does not reveal floating through N=12 and only resolves it weakly at N=18, while its stronger physical analysis uses much larger tensor-network systems. Report “floating candidate/compatible signatures” rather than a confirmed phase if restricted to small rings. [S12](S12.md), [S13](S13.md).

Chandra–Dasgupta’s perturbative diagram differs from later numerical work near the multiphase point. Use that disagreement as a warning about approximation domains, not as permission to choose whichever boundary best matches our plot. [S14](S14.md).

## 4. What does target depolarization after every CNOT do, and can every shift direction be predicted?

PennyLane uses total nonidentity-Pauli probability p:

\[
\mathcal D_p(\rho)=(1-p)\rho+\frac p3(X\rho X+Y\rho Y+Z\rho Z),\qquad a=1-4p/3.
\]

Fully mixed occurs at p=.75, not1. The challenge places this on the **target only**, immediately after each CNOT. [S03](S03.md), [S26](S26.md).

Our complete derivation and diagnostic-by-diagnostic prediction table are in [S28](S28.md). For an explicitly separate terminal-noise benchmark, m identical channels/site yield Cij′=a^(2m)Cij for i≠j, mx′=a^m mx, and S′(q)=1+a^(2m)[S(q)−1]. One terminal layer gives pair contraction 0.973511 at p=.01 and 0.871111 at p=.05. Uniform contraction leaves q*, equal-structure-factor crossings and derivative-peak locations unchanged. Fixed order-score thresholds above the noise floor can move toward lower h if the score decreases with h.

These formulas are **not a substitute for the actual interleaved circuit**. Backward propagation through CNOTs changes a Pauli string’s support; generic rotations split it into components. The first-order noise response is a sum of insertion responses whose signs depend on the ideal state and later gates. For an estimator f(h,p)=0,

\[
\delta h=-p\frac{\partial_p f(h_0,0)}{\partial_h f(h_0,0)}+O(p^2).
\]

This predicts a shift only after fixing the circuit, parameters and estimator. Reoptimization adds a parameter-response term. Therefore the request for a universal direction for **each** boundary is underdetermined: lower-h threshold shrinkage is a conditional prediction, not a theorem for all classifiers. Neither a ferro-versus-antiphase robustness ranking nor a signed floating-width change is determined solely by p. [S28](S28.md).

Also, gate noise does not modify the specified H or its exact spectrum. We measure distorted **phase diagnostics of prepared states**, not a new equilibrium Hamiltonian phase diagram. A common CPTP map increases neighboring-state Uhlmann fidelity, but differently parameterized interleaved circuits need not constitute that same common map. QCNN and autoencoder outputs require their own response analysis. [S28](S28.md).

Proposed future protocol: optimize clean parameters; fix parameters, gate ordering and depth across p; derive or evaluate ideal-state insertion responses before noisy scans; compare predictions against measured shifts with uncertainty. Separately label any noise-aware reoptimization. This proposal is not an experiment already performed.

## 5. Which mitigation methods are meaningful on default.mixed?

| Method | Meaningful use | What it cannot recover |
|---|---|---|
| Controlled-noise ZNE | Scale channel strength at every specified location; extrapolate each correlator/expectation with fixed ideal circuit and parameters | Finite-depth ansatz bias, bad minima, finite-N physics, uncontrolled higher-order extrapolation |
| Symmetry verification | Evaluate parity-sector expectation ratios using P=ΠXi and OP | Errors remaining inside the desired sector; an unknown/incorrect target sector |
| Analytic rescaling | Benchmark known terminal-channel contraction or verified special Pauli trajectories | General interleaved RY/CNOT noise through one universal a² factor |
| Probabilistic cancellation | Calibrated signed inverse-noise representation, with explicit overhead | Cost-free scalable recovery of lost information |
| Readout mitigation | Only relevant if a readout-error model is deliberately added | Gate noise in the present challenge by itself |

ZNE cancels successive terms of E(λ)=E0+a1λ+a2λ²+… using weights with Σγj=1 and Σγj cj^k=0. Extra scaled-noise samples are needed; the required three report diagrams are not automatically an adequate extrapolation design. Keep p=0 truth out of the fit and use it as a benchmark. Mitigate components first, then rebuild S(q) and apply a fixed classifier; directly extrapolating categorical labels is ill-defined. Shot-free simulation removes sampling variation but not truncation bias or extrapolation instability. [S19](S19.md), [S20](S20.md).

For symmetry verification, when [O,P]=0 and target parity is s,

\[
\langle O\rangle_s=\frac{\langle O\rangle+s\langle OP\rangle}{1+s\langle P\rangle}.
\]

All ZZ correlators and Xi commute with P. Our ground-sector argument: for h>0 the finite ZZ/X Hamiltonian has negative single-spin-flip off-diagonal elements connecting the computational-basis hypercube, giving a unique strictly positive ground state; global spin flip then has eigenvalue+1. At h=0 degeneracy invalidates that uniqueness argument. Evaluate sector acceptance, energy/observable change, and residual error; do not assume parity verification corrects every Pauli fault. [S21](S21.md); sector argument is our deduction.

The pinned 0.44.1 mixed device supports analytic backprop; parameter-shift for supported rotation-parameter expectations and finite differences are QNode transforms. Adjoint is not a mixed-device derivative. No interface/runtime validation was executed in this research step. [S25](S25.md), [S27](S27.md).

## 6. What are the VQE pitfalls near κ=.5 and h≈0?

At h=0 and N divisible by 4, classical ferro and antiphase energies per site are−1+κ and−κ, crossing at κ=.5. The multiphase point has many degenerate domain configurations. Small transverse field, small level splittings and multiple symmetry-related states make wavefunction/observable accuracy much harder than energy accuracy. A variational state can look energetically excellent while selecting the wrong ordered representative or lacking the correlations of the desired finite-ring state. [S14](S14.md), [S11](S11.md); energy comparison is our arithmetic.

The Monaco study explicitly demonstrates low-field/critical fidelity problems despite subpercent energy agreement. Use multiple starts and physical initial states, selected symmetry checks, overlap/subspace fidelity, residuals and correlation errors. Warm starts help but can perpetuate a branch or sector; forward/reverse sweeps and independent restarts diagnose this. Sparse-ED starts should contain all targeted states, not just the previous ground state when a gap is needed. [S11](S11.md), [S23](S23.md).

Increasing ansatz depth trades expressiveness against more noise. Wang proves exponential gradient suppression for its specified all-qubit local-Pauli-noise layer model, even away from random initialization. It is a warning about scaling, not proof our target-only N=8 circuit has already hit a barren plateau. Exact simulator gradients have no shot floor; finite-shot hardware does. Local costs, warm starts and parameter sharing do not overturn the theorem’s noise-induced mechanism when its assumptions hold. [S18](S18.md).

Stilck França–García-Patrón bounds compare noisy optimization energies with classical methods; they do not imply every output is literally a Gibbs state or every small VQE is useless. Their correlation-length/depth tension motivates shallow physical ansätze and explicit depth sweeps. [S22](S22.md).

Two practical source pitfalls amplify this region’s difficulty: CERN DMRG silently floors h at.1, and the starter noise ansatz uses one shared scalar angle. Neither should be treated as a validated low-field ground-state reference without modification in a later implementation task. [S06](S06.md), [S07](S07.md).

## 7. Which extensions are supported by the graders’ own references?

The rubric explicitly rewards N≥12, multiple methods, error mitigation, floating-phase detection and Trotterized dynamical signatures. The strongest ordering of effort is an inference from the rubric and source limitations, not an organiser promise of bonus points. [S01](S01.md).

1. **Establish an accurate finite-ring backbone.** Sparse/ED N=8 and 12, transparent S(q)/mx diagnostics, fidelity cuts and quantified disagreement with approximate overlays. Treat N=10 as a commensurability experiment rather than an interchangeable size.
2. **Make noise analysis quantitative.** Fixed circuit/parameters across p, extracted displacement with uncertainty, depth dependence, and comparison against the channel-response prediction. Separate preparation error from added noise.
3. **Demonstrate useful mitigation.** Compare raw and corrected correlators and boundary estimates to held-out clean truth; show remaining error and overhead. A failed but well-diagnosed recovery is scientifically informative.
4. **Compare a learned method to the physical diagnostics.** The organisers cite QCNN and autoencoder work. Check label encodings, training leakage and sensitivity to reference states. An autoencoder anomaly score alone is not an identified floating phase. [S02](S02.md), [S11](S11.md), [S12](S12.md).
5. **Add dynamics only with controlled error accounting.** The Heisenberg challenge directly motivates target-noisy Trotter circuits. Separate exact-evolution error, noiseless Trotter error and channel error; a finite-size rate-function feature is corroboration, not a substitute for required equilibrium maps. [S03](S03.md), [S04](S04.md).

For the 2–3 page write-up, keep one main phase map panel, one quantitative noise/mitigation cut, and a small boundary-shift table with uncertainty; put source audits and convergence details in the notebook/appendix. This is a proposed presentation choice, not a deliverable already produced.

## Remaining access gaps and scope

Selke (1988) and original Peschel–Emery (1981) full texts remain inaccessible. Both are explicitly labeled abstract/preview-only; detailed conclusions rely on the later full sources cited above. The full accessible Gu, Sandvik and QuSpin texts, CERN current source tree, and scoped PennyLane documentation have now been reviewed. The reading audit totals 26 FULL source items, two abstract/preview-only items and one separate original derivation. No phase diagrams, simulation results or project implementation are claimed by this literature step.
