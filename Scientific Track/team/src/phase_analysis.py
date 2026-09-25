"""Phase classification, boundary extraction and uncertainty bookkeeping (shared by notebook/figures).

Order parameters (floor-subtracted, normalised so a perfect pattern gives 1 and an uncorrelated
state gives 0; exactly the off-diagonal part of S(q) that terminal noise contracts [S28]):
    O_F = (S(0) - 1)/(N - 1),     O_A = (S(pi/2) - 1)/(N/2 - 1)       (N divisible by 4)
Labels: 0 ferro, 1 antiphase, 2 paramagnet (m_x >= T_x), 3 unpolarised/noise-dominated (neither order
nor m_x above threshold). A q* not in {0, pi/2} is reported separately as 'modulated'; it is NOT a
floating label by itself (gapped modulated paramagnets exist beyond the Peschel-Emery line [S10])."""
from __future__ import annotations

import json
import numpy as np
from scipy.interpolate import PchipInterpolator

import scan_io as sio

LABELS = ["ferro", "antiphase", "paramagnet", "unpolarised"]


# ---------------------------------------------------------------- loading

def load_scan(scan, nrows):
    metas, d, info = sio.merge_rows(scan, nrows)
    d["kappa"] = np.array([m["kappa"] for m in metas])
    d["h"] = d["h"][0]
    return metas, d, info


def orders(Sq, n):
    OF = (Sq[..., 0] - 1) / (n - 1)
    OA = (Sq[..., n // 4] - 1) / (n / 2 - 1) if n % 4 == 0 else np.full(Sq.shape[:-1], np.nan)
    return OF, OA


# ---------------------------------------------------------------- calibration

def anchor_masks(K, Hh):
    """Unambiguous interior anchors from limiting-case physics (NOT from the overlay curves):
    ferro: kappa<=0.2, h<=0.2 (classical ferro ground state, weak field); antiphase: kappa>=0.8, h<=0.1;
    paramagnet: h>=1.8 (field-dominated) for all kappa."""
    return {"ferro": (K <= 0.2 + 1e-9) & (Hh <= 0.2 + 1e-9),
            "antiphase": (K >= 0.8 - 1e-9) & (Hh <= 0.1 + 1e-9),
            "para": Hh >= 1.8 - 1e-9}


def calibrate(OF, OA, mx, K, Hh):
    a = anchor_masks(K, Hh)
    order = np.fmax(OF, OA)
    ord_anch = np.concatenate([OF[a["ferro"]], OA[a["antiphase"]]])
    para_anch = order[a["para"]]
    mx_ord = np.concatenate([mx[a["ferro"]], mx[a["antiphase"]]])
    T = 0.5 * (np.median(ord_anch) + np.median(para_anch))
    Tx = 0.5 * (np.median(mx_ord) + np.median(mx[a["para"]]))
    return {"T_order": float(T), "T_mx": float(Tx),
            "anchor_medians": {"order_ordered": float(np.median(ord_anch)), "order_para": float(np.median(para_anch)),
                               "mx_ordered": float(np.median(mx_ord)), "mx_para": float(np.median(mx[a["para"]]))},
            "anchor_counts": {k: int(v.sum()) for k, v in a.items()},
            "anchor_rule": "ferro kappa<=0.2&h<=0.2; antiphase kappa>=0.8&h<=0.1; para h>=1.8; thresholds = midpoints of anchor medians"}


def classify(OF, OA, mx, T, Tx):
    lab = np.full(OF.shape, 3, dtype=int)
    order = np.fmax(OF, OA)
    lab[(order < T) & (mx >= Tx)] = 2
    lab[(order >= T) & (OF >= OA)] = 0
    lab[(order >= T) & (OA > OF)] = 1
    return lab


def ambiguity(OF, OA, mx, T, Tx, dT=0.1, dTx=0.1):
    base = classify(OF, OA, mx, T, Tx)
    amb = np.zeros(base.shape, bool)
    for t in (T - dT, T + dT):
        for tx in (Tx - dTx, Tx + dTx):
            amb |= classify(OF, OA, mx, t, tx) != base
    return amb


def modulated(Sq, n):
    qi = np.argmax(Sq[..., : n // 2 + 1], axis=-1)
    return (qi != 0) & (qi != n // 4), qi


# ---------------------------------------------------------------- boundaries along h

def _crossing(h, y, T, kind="linear"):
    """First downward crossing of y through T along h (y(h0) >= T required)."""
    if not np.isfinite(y[0]) or y[0] < T:
        return np.nan
    idx = np.where((y[:-1] >= T) & (y[1:] < T))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if kind == "linear":
        return float(h[i] + (T - y[i]) * (h[i + 1] - h[i]) / (y[i + 1] - y[i]))
    f = PchipInterpolator(h, y - T)
    lo, hi = h[i], h[i + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _deriv_peak(h, y, need_start=None):
    """h of max(-dy/dh) using centred differences + parabolic refinement."""
    if need_start is not None and (not np.isfinite(y[0]) or y[0] < need_start):
        return np.nan
    d = -np.gradient(y, h)
    i = int(np.nanargmax(d))
    if 0 < i < len(h) - 1:
        y0, y1, y2 = d[i - 1], d[i], d[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        off = float(np.clip(off, -1, 1))
        return float(h[i] + off * (h[1] - h[0]))
    return float(h[i])


def boundaries(K, h, OF, OA, T, dT=0.1):
    """Per kappa row: which order is present at h_min, threshold crossing (linear, pchip, and over
    T +- dT), derivative peak. Returns dict of arrays over kappa."""
    out = {k: [] for k in ["phase", "h_T", "h_T_pchip", "h_T_lo", "h_T_hi", "h_D"]}
    for i in range(len(K)):
        of, oa = OF[i], OA[i]
        if max(of[0], oa[0]) < T:
            y, ph = (of, -1)
        else:
            y, ph = (of, 0) if of[0] >= oa[0] else (oa, 1)
        out["phase"].append(ph)
        if ph < 0:
            for k in ["h_T", "h_T_pchip", "h_T_lo", "h_T_hi", "h_D"]:
                out[k].append(np.nan)
            continue
        out["h_T"].append(_crossing(h, y, T))
        out["h_T_pchip"].append(_crossing(h, y, T, "pchip"))
        out["h_T_lo"].append(_crossing(h, y, T + dT))   # higher threshold -> earlier crossing
        out["h_T_hi"].append(_crossing(h, y, T - dT))
        out["h_D"].append(_deriv_peak(h, y, need_start=T))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def fidelity_peak(h, chi):
    """h (midpoint convention) of the maximum of chi_F along the sweep (chi evaluated between h_j,h_j+1)."""
    hm = 0.5 * (h[:-1] + h[1:])
    c = chi[:-1]
    if not np.any(np.isfinite(c)):
        return np.nan
    i = int(np.nanargmax(c))
    if 0 < i < len(c) - 1 and np.all(np.isfinite(c[i - 1:i + 2])):
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        den = y0 - 2 * y1 + y2
        off = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den != 0 else 0.0
        return float(hm[i] + off * (hm[1] - hm[0]))
    return float(hm[i])


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda x: float(x) if np.isscalar(x) else np.asarray(x).tolist()))


def classify_ratio(OF, OA, mx):
    """Threshold-free 'relative dominance' labels: argmax(O_F, O_A, m_x^2) -> 0/1/2.
    Motivation [S28]: under uniform terminal depolarisation O_F, O_A and m_x^2 all contract by the
    same factor a^(2m), so this label is exactly noise-invariant in that benchmark; any change under
    the real interleaved circuit therefore measures non-terminal (circuit-specific) noise action.
    Defined a priori from the channel algebra, not tuned to the overlay curves."""
    st = np.stack([OF, OA, np.square(mx)], axis=-1)
    return np.argmax(st, axis=-1)
