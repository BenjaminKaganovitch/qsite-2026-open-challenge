"""Sharded, resumable, atomically written scan storage shared by every CLI.

Layout: team/results/<scan_name>/row_<iii>.npz  (one complete h-sweep = one kappa row)
Each row file carries a JSON 'meta' string; merge_rows() validates that every row
is present exactly once and that metadata (grid, N, method, ...) is identical."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

TEAM = Path(__file__).resolve().parent.parent
RESULTS = TEAM / "results"


def set_single_thread():
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "JAX_NUM_THREADS"]:
        os.environ[v] = "1"


def grid(nk=31, nh=31, kmin=0.0, kmax=1.0, hmin=0.0, hmax=2.0):
    return np.linspace(kmin, kmax, nk), np.linspace(hmin, hmax, nh)


def parse_rows(spec: str, nrows: int):
    if spec in (None, "", "all"):
        return list(range(nrows))
    out = []
    for part in spec.split(","):
        if ":" in part:
            a, b = part.split(":")
            out += list(range(int(a or 0), int(b or nrows)))
        else:
            out.append(int(part))
    bad = [r for r in out if r < 0 or r >= nrows]
    if bad:
        raise SystemExit(f"rows out of range: {bad}")
    return out


def row_path(scan: str, row: int) -> Path:
    return RESULTS / scan / f"row_{row:03d}.npz"


def row_done(scan: str, row: int, meta_core: dict) -> bool:
    p = row_path(scan, row)
    if not p.exists():
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            m = json.loads(str(z["meta"]))
        return all(m.get(k) == v for k, v in meta_core.items())
    except Exception:
        return False


def save_row(scan: str, row: int, meta: dict, arrays: dict):
    d = RESULTS / scan
    d.mkdir(parents=True, exist_ok=True)
    final = row_path(scan, row)
    tmp = d / f".tmp_row_{row:03d}_{os.getpid()}.npz"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, meta=json.dumps(meta, default=float), **arrays)
    os.replace(tmp, final)  # atomic on POSIX and NTFS


def merge_rows(scan: str, nrows: int, compare_keys=None):
    """Load all rows, validate completeness/uniqueness/metadata compatibility, stack arrays."""
    d = RESULTS / scan
    files = sorted(d.glob("row_*.npz"))
    rows = [int(f.stem.split("_")[1]) for f in files]
    missing = sorted(set(range(nrows)) - set(rows))
    dup = sorted({r for r in rows if rows.count(r) > 1})
    extra = sorted(set(rows) - set(range(nrows)))
    if missing or dup or extra:
        raise ValueError(f"{scan}: missing rows {missing}, duplicated {dup}, out-of-range {extra}")
    metas, data = [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            metas.append(json.loads(str(z["meta"])))
            data.append({k: z[k] for k in z.files if k != "meta"})
    keys = compare_keys or [k for k in metas[0] if k not in ("row", "kappa", "timestamp", "runtime_s", "host", "versions", "peak_rss_mb")]
    for m in metas[1:]:
        for k in keys:
            if m.get(k) != metas[0].get(k):
                raise ValueError(f"{scan}: incompatible metadata key {k}: {m.get(k)} vs {metas[0].get(k)}")
    vers = {json.dumps(m.get("versions"), sort_keys=True) for m in metas}
    stacked = {k: np.stack([dd[k] for dd in data]) for k in data[0]}
    return metas, stacked, {"n_version_sets": len(vers)}


def boundary_discontinuity_report(arr2d: np.ndarray, label: str, factor=6.0):
    """Flag kappa rows whose change to the next row is anomalously large compared with the
    median row-to-row change (shards are rows, so this also checks shard boundaries)."""
    diff = np.nanmax(np.abs(np.diff(arr2d, axis=0)), axis=1)
    med = np.nanmedian(diff) + 1e-15
    flagged = [int(i) for i in np.where(diff > factor * med)[0]]
    return {"observable": label, "median_row_jump": float(med), "max_row_jump": float(np.nanmax(diff)),
            "flagged_row_pairs": flagged}


def peak_rss_mb():
    """Peak resident memory of this process in MiB (Linux/macOS/Windows)."""
    import sys
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2**20
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024.0


class Progress:
    def __init__(self, total, label):
        self.total, self.label, self.t0, self.done = total, label, time.time(), 0

    def step(self, msg=""):
        self.done += 1
        el = time.time() - self.t0
        eta = el / self.done * (self.total - self.done)
        print(f"[{self.label}] {self.done}/{self.total}  elapsed {el:6.1f}s  eta {eta:6.1f}s  {msg}", flush=True)
