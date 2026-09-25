"""Exact first-order channel-insertion responses [S28] resolved by layer and bond type at
representative points (fixed clean-optimised HVA parameters). CLI; writes
team/results/analysis/insertion_response_N8_L4.json.
d<O>/dp|_0 = sum_g Tr{O_g L_t(rho_g)}; each location's term is obtained exactly (to O(dp^2)) by a
central difference with the channel switched on at that location only."""
from scan_io import set_single_thread
set_single_thread()
import json
import numpy as np
import annni_core as ac
import hva
import phase_analysis as pa
import scan_io as sio

N, L = 8, 4
POINTS = {"ferro (0.2,0.2)": (0.2, 0.2), "ferro near boundary (0.2,0.6)": (0.2, 0.6),
          "antiphase (0.8,0.2)": (0.8, 0.2), "antiphase near boundary (0.8,0.6)": (0.8, 0.6),
          "paramagnet (0.3,1.5)": (0.3, 1.5), "paramagnet (0.8,1.5)": (0.8, 1.5)}


def obs_vec(rho):
    o = ac.observables_rho(N, rho)
    OF, OA = pa.orders(o["Sq"][None], N)
    return np.array([OF[0], OA[0], o["mx"]])


def main():
    import sys
    sel = [int(a) for a in sys.argv[1:]] or list(range(len(POINTS)))
    ks, hs = sio.grid()
    ops = hva.gate_list(N, L)
    # location metadata: layer and bond class for each CNOT (two CNOTs per ZZ bond)
    loc = []
    nn, nnn = ac.bonds(N)
    for l in range(L):
        for cls, bl in (("NN", nn), ("NNN", nnn)):
            for b in bl:
                loc += [(l, cls, "first"), (l, cls, "second")]
    assert len(loc) == hva.n_cnots(N, L)
    out = {}
    dp = 1e-4
    for pidx, (name, (kv, hv)) in enumerate(POINTS.items()):
        dest = sio.RESULTS / 'analysis' / 'insertion' / f'point_{pidx}.json'
        if pidx not in sel or dest.exists():
            continue
        i, j = int(np.argmin(abs(ks - kv))), int(np.argmin(abs(hs - hv)))
        with np.load(sio.row_path("vqe_hva_N8_L4_31x31", i)) as z:
            g = z["params"][j]
        base = obs_vec(hva.gate_density_matrix(g, N, L, 0.0))
        per = []
        for gidx in range(len(loc)):
            pl = np.zeros(len(loc)); pl[gidx] = dp
            plus = obs_vec(hva.gate_density_matrix(g, N, L, 0.0, pl))
            pl[gidx] = -dp
            minus = obs_vec(hva.gate_density_matrix(g, N, L, 0.0, pl))
            per.append((plus - minus) / (2 * dp))
        per = np.array(per)
        tot = per.sum(0)
        by_layer = {f"layer{l+1}": per[[k for k, t in enumerate(loc) if t[0] == l]].sum(0).tolist() for l in range(L)}
        by_cls = {c: per[[k for k, t in enumerate(loc) if t[1] == c]].sum(0).tolist() for c in ("NN", "NNN")}
        by_pos = {c: per[[k for k, t in enumerate(loc) if t[2] == c]].sum(0).tolist() for c in ("first", "second")}
        # terminal-equivalent benchmark: 16 target hits per site -> d/dp of a^(2m) O = -(4/3)(2m) O at p=0
        m = 4 * L
        out[name] = {"kappa": float(ks[i]), "h": float(hs[j]), "O_F,O_A,m_x (p=0)": base.tolist(),
                     "dO/dp total [O_F,O_A,m_x]": tot.tolist(),
                     "relative dlnO/dp": (tot / np.where(np.abs(base) > 1e-9, base, np.nan)).tolist(),
                     "terminal-equivalent dlnO/dp (pair: -8m/3, m_x: -4m/3), m=16": [-8 * m / 3, -8 * m / 3, -4 * m / 3],
                     "by_layer": by_layer, "by_bond_class": by_cls, "by_cnot_position": by_pos}
        pa.save_json(dest, {name: out[name]})
        print(name, np.round(tot, 3), np.round(out[name]["relative dlnO/dp"], 2), {k: np.round(v, 2).tolist() for k, v in by_layer.items()})
    parts = sorted((sio.RESULTS / 'analysis' / 'insertion').glob('point_*.json'))
    if len(parts) == len(POINTS):
        merged = {}
        for f in parts:
            merged.update(json.loads(f.read_text()))
        pa.save_json(sio.RESULTS / "analysis" / "insertion_response_N8_L4.json", merged)
        print("merged", len(merged))


if __name__ == "__main__":
    main()
