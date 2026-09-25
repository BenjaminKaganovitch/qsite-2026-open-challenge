# Team workspace: Q-SITE Hacks 2026 (Quantum Coalition Open Challenge)

Upstream starter kit: https://github.com/benmcdonough20/QSITE-2026-QuantumCoalition (MIT). Do not edit files under `starter_kit/`; pull fixes with `git pull upstream main`.

## Roles
| Member | Focus |
|---|---|
| Benjamin | Lead, SDK architecture, integration, repo |
| Dimitri | Theory, Hamiltonian conventions, analytical boundaries, validation |
| Yan | Circuits, noise model, error mitigation |
| Frederick | Classical baselines (ED/DMRG), tooling, plots, write-up |

## Layout
Our work lives in `<Track>/team/`: `notebooks/` (one owner per notebook), `src/` (shared .py modules), `results/` (figures, .npz data), `writeup/`.

## Git rules
- `main` always runs. Work on short-lived branches: `<name>/<topic>`, merge via PR.
- Shared logic goes in `team/src/*.py`, not copy-pasted between notebooks.
- Clear notebook outputs before committing (or install `nbstripout`).
- No coding-challenge solutions in this repo: it goes public on Sept 25.

## Deadlines (EDT)
- Sept 23, 11:59 PM: project draft
- Sept 25, 11:59 PM: final submission (repo public, write-up, video)

## Citations
Log every paper, demo, and tool used here as you go (required by the rules).
- PennyLane ANNNI demo: https://pennylane.ai/qml/demos/tutorial_annni
- PennyLane Noisy Heisenberg challenge: https://pennylane.ai/challenges/heisenberg_model
