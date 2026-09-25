# Scientific Track: ANNNI phase diagram under noise (team workspace)

## Deliverables
| Item | Path |
|---|---|
| Notebook (regenerates every figure from cached arrays) | `notebooks/ANNNI_phase_diagram_under_noise.ipynb` |
| Required diagrams, p = 0 / 0.01 / 0.05 (PNG + PDF) | `results/figures/phase_diagram_p0.0.*`, `phase_diagram_p0.01.*`, `phase_diagram_p0.05.*` |
| Independent clean ED reference diagram | `results/figures/fig_ed_phase_N8.png`, `results/figures/fig_phase_diagrams_all.png` |
| Report (2–3 pages) | `writeup/report.md` |
| Slides and speaker notes | `writeup/slides.pdf`, `writeup/presentation_notes.md` |
| Raw data (one `.npz` per κ row, metadata embedded) | `results/<scan_name>/row_iii.npz` |
| Analysis tables and JSON summaries | `results/analysis/` |
| Validation reports | `results/validation/` |
| Literature notes (`[Sxx]` citations) | `literature/` |

## Conventions
$H=-\sum Z_iZ_{i+1}+\kappa\sum Z_iZ_{i+2}-h\sum X_i$, with Pauli operators, $J_1=1$ and periodic boundaries. Wire 0 is the most significant bit (PennyLane ordering). $S(q)=\frac1N\sum_{ij}e^{iq(i-j)}\langle Z_iZ_j\rangle$ with the self-terms kept. Noise is `qml.DepolarizingChannel(p)` on the target qubit immediately after every CNOT, simulated on `default.mixed`. All noisy runs use the fixed-parameter protocol: parameters are optimised at p=0 and reused for every p.

## Environment (Python 3.14, PennyLane 0.44.1, uv)
The organiser's `../pyproject.toml` is left unchanged. `team/pyproject.toml` adds the dependencies it lacks (SciPy, nbconvert, ipykernel), and `uv.lock` pins the exact versions.

**macOS / Linux (bash):**
```bash
cd "Scientific Track/team"
uv python install 3.14
uv sync --python 3.14
.venv/bin/python src/validate_core.py && .venv/bin/python src/validate_circuits.py
```
**Windows 11 (PowerShell):**
```powershell
cd "C:\path\to\qsite-open\Scientific Track\team"
uv python install 3.14
uv sync --python 3.14
.\.venv\Scripts\python.exe src\validate_core.py; .\.venv\Scripts\python.exe src\validate_circuits.py
```
Every CLI sets the BLAS/OpenMP thread counts to 1 before importing NumPy.

## Scans (sharded, resumable, atomic)
Every scan CLI takes `--rows a:b` (or a comma list) and writes `results/<scan>/row_iii.npz` atomically, so a rerun skips rows that are already complete. The noisy CLIs also checkpoint after every point. Merging (`scan_io.merge_rows`) checks for missing, duplicated or out-of-range rows and for incompatible metadata.

| Step | Command (run from `team/`) | Measured cost (1 core) |
|---|---|---|
| Clean ED, N=8/10/12, 31×31 | `python src/ed_scan.py --n 8` (or `--n 10`, `--n 12`) | 10 s / 30 s / 90 s |
| Fine ED cuts | `python src/ed_scan.py --n 12 --nk 21 --nh 401 --hmin 0.005` | ~40 s per row at N=12 |
| Clean HVA VQE, N=8, L=4 | `python src/vqe_scan.py --rows 0:31` | 18–54 s per row |
| Noisy evaluation (default.mixed) | `python src/noisy_eval.py --mode pennylane` | ~100 s per row (1.6 s per circuit) |
| ZNE points and first-order response | `python src/noisy_eval.py --mode extras` | ~80 s per row |
| Seed check | `python src/vqe_scan.py --seed 777 --scan vqe_hva_N8_L4_31x31_seed777 --rows 3,6,9,12,18,21,24,27` then `python src/noisy_eval.py --mode numpy --source vqe_hva_N8_L4_31x31_seed777 --rows ...` | |
| Channel-insertion response | `python src/insertion_response.py` | ~80 s per point |
| Figures and tables | `python src/fig_clean.py; python src/fig_noise.py; python src/fig_mitigation.py; python src/fig_finite_size.py; python src/fig_seed.py` | < 1 min |

Peak memory is under 200 MB per worker for N=8 density matrices; an N=12 density matrix alone is 256 MiB. Parallel workers (PowerShell):
```powershell
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
$p = foreach ($s in "0:8","8:16","16:24","24:31") { Start-Process -PassThru -NoNewWindow -FilePath $py -ArgumentList "src\noisy_eval.py","--mode","pennylane","--rows",$s }
$p | Wait-Process
```
bash equivalent: `for s in 0:8 8:16 16:24 24:31; do .venv/bin/python src/noisy_eval.py --mode pennylane --rows $s & done; wait`.

## Module map
`annni_core.py` (Hamiltonian, observables, overlays), `scan_io.py` (shards, merge checks), `circuits.py` (circuit engines and the RY/CNOT ansatz), `hva.py` (HVA ansatz, gate compilation, noisy density matrix, QNode), `vqe.py` (optimiser helpers), `phase_analysis.py` (calibration, classification, boundaries), `fig_*.py` (figures and tables), `make_notebook.py`, `make_slides.py`.

## Attribution
Team: Benjamin, Dimitri, Yan, Frederick. AI coding assistants were used, as permitted by the organisers. External sources are cited in `literature/LITERATURE_INDEX.md`.
