# Literature index and reading audit

Q-SITE2026 Scientific Open Challenge — literature-only pass, 2026-09-25.

**Status: literature base complete, with two explicitly documented full-text access gaps.** All accessible sources in the inventory have been read in the stated scope. The synthesis answers all seven requested questions and distinguishes published findings, source-code behavior and our derivations.

Start with [What the literature says](WHAT_THE_LITERATURE_SAYS.md), then [our channel derivation](S28.md). No project code was written; the starter kit and Computational Track were not edited. The pre-existing handoff was read, not rewritten.

## Read-status counts

**26 source items read in full; 2 abstract/preview-only; 0 partial.** One additional original derivation (S28) is not counted as an external source. Count units are listed source items: a demo, paper, code audit or explicitly scoped documentation implementation; repeated preprint/journal versions are not double-counted. FULL for a paper means the identified preprint including its appended supplement, not an uninspected publisher revision. FULL for S25/S26 means the pinned device implementation/channel class, not the entire documentation site.

The two access failures are Selke1988 and Peschel–Emery1981. Downloading a PDF is not counted as reading it.

## Fixed citation format

Use `[Sxx]` in prose, with a local note link when working in Markdown. Bibliography format: **Author(s), “Title,” venue volume, pages/article (year), DOI; exact preprint/version if that was read.** For software: owner, title, pinned commit/version where available, URL, access date. Each source note carries its complete citation and read scope.

## Source inventory

| Key | Tier | Source/note | Status |
|---|---|---|---|
| S01 | Context | [Organiser handout](S01.md) | FULL |
| S02 | 1 | [PennyLane ANNNI phase detection demo](S02.md) | FULL |
| S03 | 1 | [Noisy Heisenberg challenge](S03.md) | FULL |
| S04 | 1 | [Seeing quantum phase transitions](S04.md) | FULL |
| S05 | 1 | [Building spin Hamiltonians](S05.md) | FULL |
| S06 | 1 | [CERN reference repository audit](S06.md) | FULL |
| S07 | 1 | [Read-only starter kit audit](S07.md) | FULL |
| S08 | 2 | [Selke ANNNI review](S08.md) | ABSTRACT_ONLY |
| S09 | 2 | [Peschel–Emery and the disorder line](S09.md) | ABSTRACT_ONLY |
| S10 | 2 | [DMRG low-frustration and disorder line](S10.md) | FULL |
| S11 | 2 | [QCNN and ANNNI VQE paper](S11.md) | FULL |
| S12 | 2 | [ANNNI floating phase and autoencoder study](S12.md) | FULL |
| S13 | 2 | [DMRG high-frustration phase diagram](S13.md) | FULL |
| S14 | 2 | [Perturbative floating phase](S14.md) | FULL |
| S15 | 2 | [Ground-state overlap as a transition diagnostic](S15.md) | FULL |
| S16 | 2 | [Fidelity susceptibility review](S16.md) | FULL |
| S17 | 2 | [Critical entanglement entropy](S17.md) | FULL |
| S18 | 3 | [Noise-induced barren plateaus](S18.md) | FULL |
| S19 | 3 | [Zero-noise extrapolation and cancellation](S19.md) | FULL |
| S20 | 3 | [Variational simulation and active error reduction](S20.md) | FULL |
| S21 | 3 | [Symmetry verification](S21.md) | FULL |
| S22 | 3 | [Limits of noisy optimization](S22.md) | FULL |
| S23 | 4 | [Sandvik: Lanczos and structure factor](S23.md) | FULL |
| S24 | 4 | [QuSpin sparse construction](S24.md) | FULL |
| S25 | 4 | [PennyLane default.mixed](S25.md) | FULL |
| S26 | 4 | [PennyLane DepolarizingChannel](S26.md) | FULL |
| S27 | 4 | [PennyLane mixed-state differentiation](S27.md) | FULL |
| S29 | Context | [Team handoff](S29.md) | FULL |
| S30 | Impl. | [Hamiltonian variational ansatz (added during implementation)](S30.md) | SCOPED (abstract + introduction) |

[S28 — our depolarization derivation](S28.md) serves Tier3; S23 also supplies the standard structure-factor convention requested in Tier2.

## Most consequential findings

- Lower antiphase–floating line is PT; the upper line is KT/BKT. The organiser’s lower “BKT” label is misleading.
- The three overlay curves are approximations/fits. The Peschel–Emery disorder line is a separate curve inside the paramagnet.
- Demo/CERN use open XX/Z chains by default; starter uses periodic ZZ/X. Hadamards fix axes, not boundary conditions.
- The ANNNI demo prepares ED states; its paper contains VQE. CERN DMRG floors requested low fields at 0.1. Archived VQE data exist but require version, sign and boundary conversion.
- Correlation-based order avoids finite-size one-point symmetry cancellation; the starter antiphase proxy is insufficient.
- Uniform terminal noise yields exact damping, but the actual interleaved CNOT-target circuit has no universal boundary-shift direction.
- No numerical offset to thermodynamic curves or robust floating-phase claim at N8–12 is established by the reviewed sources.

## Explicit access and validation limits

Selke (1988) and Peschel–Emery (1981) remain abstract/publisher-preview only after full-text retrieval attempts. Their unread equations are not paraphrased from memory. The original attribution of the second-order Ising approximation is indirect through BCF (2006); see S09–S10. These are the only remaining full-text access gaps.

The CERN current tree has been fully reviewed. Its historical Zenodo v1.0.0 archive contains saved VQEs, including periodic N=6,8,10 data; see [S06](S06.md) for formats and the negative-K convention, and [the archive inventory](CERN_ARCHIVE_INVENTORY.json). Archive inspection does not establish numerical accuracy or compatibility with the challenge environment. No project simulations, model training or runtime validation were performed.

The public Heisenberg code was recovered from the page’s embedded payload after the initial plain-text extraction showed “Loading…”. Its full status includes the starter and public helper, not a nonexistent supplied completed solution.

## Provenance and reproducibility

[FETCH_MANIFEST.json](FETCH_MANIFEST.json) records downloaded source URLs and SHA256 hashes. Downloads and extracted reading text live under `/tmp/qsite-literature` and may be temporary; the durable deliverables are the notes, synthesis and manifest. PDFs are not redistributed here. The live PennyLane demo sources were read as fetched, so their hashes distinguish them from future main-branch changes.

Source retrieval, convention audit, algebra and these notes were prepared with an AI coding assistant. Claims are not independently certified; source/version and unresolved status are retained for human review.
