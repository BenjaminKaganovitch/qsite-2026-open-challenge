from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace

import networkx as nx

INF = float("inf")


@dataclass(frozen=True)
class Config:
    time_budget: float = 10.0
    max_iters: int | None = None
    seed: int = 0
    K: int = 6
    w: float = 1.0
    gamma: float = 0.7
    fut_depth: float = 0.0
    T0: float = 2.0
    decay: float = 0.995
    Tmin: float = 0.05
    sa_cycle: int = 600
    restart_mode: str = "best"
    reverse_rounds: int = 0
    embed_budget: int = 200_000
    pilot: int = 1
    pilot_frac: float = 0.2
    elite: int = 48
    pilot_mode: int = 2
    pilot_horizon: int = 2
    pilot_trigger: int = 0
    r2_sa: bool = False
    r2_cycle: int = 150
    r2_T0: float = 1.0


@dataclass
class Device:
    nodes: list
    index: dict
    n: int
    adj: list
    deg: list
    D: list
    paths: list
    edges: set = field(default_factory=set)


_DEVICE_CACHE: dict = {}


def device_for(graph: nx.Graph) -> Device:
    key = (tuple(sorted(graph.nodes)), frozenset(frozenset(e)
           for e in graph.edges))
    dev = _DEVICE_CACHE.get(key)
    if dev is not None:
        return dev
    nodes = sorted(graph.nodes)
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj = [sorted(index[u] for u in graph.neighbors(v)) for v in nodes]
    deg = [len(a) for a in adj]
    D = [[INF] * n for _ in range(n)]
    for s in range(n):
        D[s][s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if D[s][v] == INF:
                        D[s][v] = d
                        nxt.append(v)
            frontier = nxt
    paths = [[None] * n for _ in range(n)]

    def all_paths(a, b):
        if paths[a][b] is not None:
            return paths[a][b]
        if a == b:
            res = [(a,)]
        else:
            res = []
            for v in adj[a]:
                if D[v][b] == D[a][b] - 1:
                    res.extend((a,) + p for p in all_paths(v, b))
        paths[a][b] = res
        return res

    for a in range(n):
        for b in range(n):
            if D[a][b] < INF:
                all_paths(a, b)
    edges = {(a, b) for a in range(n) for b in adj[a]}
    dev = Device(nodes, index, n, adj, deg, D, paths, edges)
    _DEVICE_CACHE[key] = dev
    return dev


@dataclass
class Prog:
    logicals: list
    lindex: dict
    ops: list
    gates: list
    m: int
    d0: int
    pre1q: list = None
    post1q: list = None


def prep_program(program) -> Prog:
    logicals = sorted({q for op in program for q in op[1:]})
    lindex = {q: i for i, q in enumerate(logicals)}
    ops, gates = [], []
    pre1q, cur1q = [], []
    last = [0] * len(logicals)
    d0 = 0
    for op in program:
        if op[0] == "2Q":
            a, b = lindex[op[1]], lindex[op[2]]
            ops.append(("2Q", a, b))
            gates.append((a, b))
            pre1q.append(cur1q)
            cur1q = []
            layer = 1 + max(last[a], last[b])
            last[a] = last[b] = layer
            d0 = max(d0, layer)
        else:
            ops.append(("1Q", lindex[op[1]]))
            cur1q.append(lindex[op[1]])
    return Prog(logicals, lindex, ops, gates, len(logicals), d0, pre1q, cur1q)


def _cands(dev, gates, gi, pos, last, depth, cfg, gpow, want_all):
    D, paths = dev.D, dev.paths
    K, w, fd = cfg.K, cfg.w, cfg.fut_depth
    a, b = gates[gi]
    pa, pb = pos[a], pos[b]
    L = D[pa][pb]
    best = INF
    best_c = None
    allc = [] if want_all else None
    fut = gates[gi + 1: gi + 1 + K]
    for P in paths[pa][pb]:
        pidx = {v: i for i, v in enumerate(P)}
        la = [0] * (L + 1)
        cur = last[P[0]]
        la[0] = cur
        for i in range(1, L):
            cur = 1 + (cur if cur > last[P[i]] else last[P[i]])
            la[i] = cur
        lb = [0] * (L + 1)
        cur = last[P[L]]
        lb[L] = cur
        for j in range(L - 1, 0, -1):
            cur = 1 + (cur if cur > last[P[j]] else last[P[j]])
            lb[j] = cur
        for k in range(L):
            x, y = la[k], lb[k + 1]
            gl = 1 + (x if x > y else y)
            c = (L - 1) + 0.5 * (gl - depth if gl > depth else 0)
            if fd and fut:
                lastp = {P[k]: gl, P[k + 1]: gl}
                for i in range(1, k + 1):
                    lastp[P[i - 1]] = la[i]
                for j in range(k + 2, L + 1):
                    lastp[P[j]] = lb[j - 1]
                dep2 = gl if gl > depth else depth
            if w and fut:
                f = 0.0
                fl = 0.0
                for t, (u, v) in enumerate(fut):
                    if u == a:
                        pu = P[k]
                    elif u == b:
                        pu = P[k + 1]
                    else:
                        pu = pos[u]
                        i = pidx.get(pu)
                        if i is not None:
                            if 1 <= i <= k:
                                pu = P[i - 1]
                            elif k + 1 <= i <= L - 1:
                                pu = P[i + 1]
                    if v == a:
                        pv = P[k]
                    elif v == b:
                        pv = P[k + 1]
                    else:
                        pv = pos[v]
                        i = pidx.get(pv)
                        if i is not None:
                            if 1 <= i <= k:
                                pv = P[i - 1]
                            elif k + 1 <= i <= L - 1:
                                pv = P[i + 1]
                    dd = D[pu][pv] - 1
                    if dd:
                        f += gpow[t] * dd
                    if fd:
                        x1 = lastp.get(pu, last[pu])
                        y1 = lastp.get(pv, last[pv])
                        est = (x1 if x1 > y1 else y1) + (dd + 1) // 2 + 1
                        if est > dep2:
                            fl += gpow[t] * (est - dep2)
                c += w * (f + fd * 0.5 * fl)
            if want_all:
                allc.append((c, P, k))
            elif c < best:
                best = c
                best_c = (c, P, k)
    return allc if want_all else best_c


def _apply(P, k, L, pos, occ, last, depth, out):
    for i in range(k):
        p, q = P[i], P[i + 1]
        layer = 1 + (last[p] if last[p] > last[q] else last[q])
        last[p] = last[q] = layer
        if layer > depth:
            depth = layer
        lp, lq = occ[p], occ[q]
        occ[p], occ[q] = lq, lp
        if lp >= 0:
            pos[lp] = q
        if lq >= 0:
            pos[lq] = p
        if out is not None:
            out.append(("SWAP", p, q))
    for j in range(L, k + 1, -1):
        p, q = P[j], P[j - 1]
        layer = 1 + (last[p] if last[p] > last[q] else last[q])
        last[p] = last[q] = layer
        if layer > depth:
            depth = layer
        lp, lq = occ[p], occ[q]
        occ[p], occ[q] = lq, lp
        if lp >= 0:
            pos[lp] = q
        if lq >= 0:
            pos[lq] = p
        if out is not None:
            out.append(("SWAP", p, q))
    return depth


def _run(dev, prog, cfg, pos, occ, last, depth, swaps, g0, gpow, out=None, pilot=0, choices=None, cnt=None):
    D = dev.D
    gates = prog.gates
    ng = len(gates)
    pre = prog.pre1q
    for gi in range(g0, ng):
        if out is not None:
            for q in pre[gi]:
                out.append(("1Q", pos[q]))
        a, b = gates[gi]
        L = D[pos[a]][pos[b]]
        if L > 1:
            if pilot:
                cl = _cands(dev, gates, gi, pos, last, depth, cfg, gpow, True)
                cl.sort(key=lambda t: t[0])
                best, bc = INF, None
                for c, P, k in cl[:pilot]:
                    p2, o2, l2 = pos[:], occ[:], last[:]
                    d2 = _apply(P, k, L, p2, o2, l2, depth, None)
                    ua, ub = p2[a], p2[b]
                    lay = 1 + (l2[ua] if l2[ua] > l2[ub] else l2[ub])
                    l2[ua] = l2[ub] = lay
                    if lay > d2:
                        d2 = lay
                    if cnt is not None:
                        cnt[0] += (ng - gi) / ng
                    s2, d2 = _run(dev, prog, cfg, p2, o2, l2, d2,
                                  swaps + L - 1, gi + 1, gpow)
                    sc = s2 + 0.5 * d2
                    if sc < best - 1e-12:
                        best, bc = sc, (P, k)
                P, k = bc
            else:
                _, P, k = _cands(dev, gates, gi, pos, last,
                                 depth, cfg, gpow, False)
            if choices is not None:
                choices.append((gi, P, k))
            depth = _apply(P, k, L, pos, occ, last, depth, out)
            swaps += L - 1
        pa, pb = pos[a], pos[b]
        layer = 1 + (last[pa] if last[pa] > last[pb] else last[pb])
        last[pa] = last[pb] = layer
        if layer > depth:
            depth = layer
        if out is not None:
            out.append(("2Q", pa, pb))
    if out is not None and g0 == 0:
        for q in prog.post1q:
            out.append(("1Q", pos[q]))
    return swaps, depth


def _swap1(p, q, pos, occ, last, depth):
    layer = 1 + (last[p] if last[p] > last[q] else last[q])
    last[p] = last[q] = layer
    lp, lq = occ[p], occ[q]
    occ[p], occ[q] = lq, lp
    if lp >= 0:
        pos[lp] = q
    if lq >= 0:
        pos[lq] = p
    return layer if layer > depth else depth


def route_swap_pilot(dev, prog, pos0, cfg, horizon_gates=2, emit=False, cnt=None, max_steps=None):
    D = dev.D
    n = dev.n
    gates = prog.gates
    ng = len(gates)
    gpow = [cfg.gamma ** t for t in range(cfg.K + 1)]
    pos = list(pos0)
    occ = [-1] * n
    for l, p in enumerate(pos):
        occ[p] = l
    last = [0] * n
    depth = 0
    swaps = 0
    out = [] if emit else None
    adj = dev.adj
    steps = 0
    max_steps = max_steps if max_steps is not None else 4 * ng * dev.n
    gi = 0
    flushed = -1
    while gi < ng:
        if out is not None and flushed < gi:
            for q in prog.pre1q[gi]:
                out.append(("1Q", pos[q]))
            flushed = gi
        a, b = gates[gi]
        L = D[pos[a]][pos[b]]
        if L > 1 and steps < max_steps:
            steps += 1
            _, P, k = _cands(dev, gates, gi, pos, last,
                             depth, cfg, gpow, False)
            p2, o2, l2 = pos[:], occ[:], last[:]
            d2 = _apply(P, k, L, p2, o2, l2, depth, None)
            ua, ub = p2[a], p2[b]
            lay = 1 + (l2[ua] if l2[ua] > l2[ub] else l2[ub])
            l2[ua] = l2[ub] = lay
            if lay > d2:
                d2 = lay
            if cnt is not None:
                cnt[0] += (ng - gi) / ng
            s2, d2 = _run(dev, prog, cfg, p2, o2, l2, d2,
                          swaps + L - 1, gi + 1, gpow)
            best, choice = s2 + 0.5 * d2, None
            qs = {a, b}
            for (u, v) in gates[gi + 1: gi + 1 + horizon_gates]:
                qs.add(u)
                qs.add(v)
            edges = set()
            for q in qs:
                p = pos[q]
                for r in adj[p]:
                    edges.add((p, r) if p < r else (r, p))
            for (p, r) in sorted(edges):
                p2, o2, l2 = pos[:], occ[:], last[:]
                d2 = _swap1(p, r, p2, o2, l2, depth)
                if cnt is not None:
                    cnt[0] += (ng - gi) / ng
                s2, d2 = _run(dev, prog, cfg, p2, o2, l2,
                              d2, swaps + 1, gi, gpow)
                sc = s2 + 0.5 * d2
                if sc < best - 1e-12:
                    best, choice = sc, (p, r)
            if choice is not None:
                p, r = choice
                depth = _swap1(p, r, pos, occ, last, depth)
                swaps += 1
                if out is not None:
                    out.append(("SWAP", p, r))
                continue
            depth = _apply(P, k, L, pos, occ, last, depth, out)
            swaps += L - 1
        elif L > 1:
            _, P, k = _cands(dev, gates, gi, pos, last,
                             depth, cfg, gpow, False)
            depth = _apply(P, k, L, pos, occ, last, depth, out)
            swaps += L - 1
        pa, pb = pos[a], pos[b]
        layer = 1 + (last[pa] if last[pa] > last[pb] else last[pb])
        last[pa] = last[pb] = layer
        if layer > depth:
            depth = layer
        if out is not None:
            out.append(("2Q", pa, pb))
        gi += 1
    if out is not None:
        for q in prog.post1q:
            out.append(("1Q", pos[q]))
    return swaps + 0.5 * depth, pos, out


def route(dev: Device, prog: Prog, pos0: list, cfg: Config, emit: bool = False, pilot: int = 0, cnt=None):
    n = dev.n
    pos = list(pos0)
    occ = [-1] * n
    for l, p in enumerate(pos):
        occ[p] = l
    last = [0] * n
    gpow = [cfg.gamma ** t for t in range(cfg.K + 1)]
    out = [] if emit else None
    swaps, depth = _run(dev, prog, cfg, pos, occ, last, 0,
                        0, 0, gpow, out, pilot, None, cnt)
    return swaps + 0.5 * depth, pos, out


def interaction_graph(prog: Prog):
    wts: dict = {}
    for a, b in prog.gates:
        e = (a, b) if a < b else (b, a)
        wts[e] = wts.get(e, 0) + 1
    nbrs = [set() for _ in range(prog.m)]
    for a, b in wts:
        nbrs[a].add(b)
        nbrs[b].add(a)
    return wts, nbrs


def find_embedding(dev: Device, prog: Prog, budget: int):
    wts, nbrs = interaction_graph(prog)
    m = prog.m
    if m > dev.n or len(wts) > len(dev.edges) // 2:
        return None
    if max((len(s) for s in nbrs), default=0) > max(dev.deg):
        return None
    order, seen = [], set()
    for start in sorted(range(m), key=lambda x: -len(nbrs[x])):
        if start in seen:
            continue
        seen.add(start)
        queue = [start]
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in sorted(nbrs[u], key=lambda x: -len(nbrs[x])):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
    pos = [-1] * m
    used = [False] * dev.n
    adjset = [set(a) for a in dev.adj]
    count = [0]

    def rec(i):
        if i == m:
            return True
        count[0] += 1
        if count[0] > budget:
            return False
        u = order[i]
        placed = [pos[v] for v in nbrs[u] if pos[v] >= 0]
        if placed:
            cands = [p for p in dev.adj[placed[0]] if not used[p]]
        else:
            cands = [p for p in range(dev.n) if not used[p]]
        du = len(nbrs[u])
        for p in cands:
            if dev.deg[p] < du:
                continue
            if any(q not in adjset[p] for q in placed[1:]):
                continue
            pos[u] = p
            used[p] = True
            if rec(i + 1):
                return True
            pos[u] = -1
            used[p] = False
            if count[0] > budget:
                return False
        return False

    return list(pos) if rec(0) else None


def greedy_placement(dev: Device, prog: Prog, rng: random.Random, start_phys=None):
    wts, nbrs = interaction_graph(prog)
    m, n, D = prog.m, dev.n, dev.D
    W = [[0] * m for _ in range(m)]
    for (a, b), c in wts.items():
        W[a][b] = W[b][a] = c
    strength = [sum(r) for r in W]
    pos = [-1] * m
    used = [False] * n
    first = max(range(m), key=lambda x: (strength[x], -x))
    if start_phys is None:
        ecc = [max(r) for r in D]
        start_phys = min(
            range(n), key=lambda p: (-dev.deg[p], sum(D[p]), ecc[p]))
    pos[first] = start_phys
    used[start_phys] = True
    placed = [first]
    while len(placed) < m:
        rest = [x for x in range(m) if pos[x] < 0]
        nxt = max(rest, key=lambda x: (
            sum(W[x][y] for y in placed), strength[x], -x))
        best, bp = INF, None
        for p in range(n):
            if used[p]:
                continue
            c = sum(W[nxt][y] * D[p][pos[y]] for y in placed)
            c += 1e-3 * rng.random()
            if c < best:
                best, bp = c, p
        pos[nxt] = bp
        used[bp] = True
        placed.append(nxt)
    return pos


def reverse_refine(dev: Device, prog: Prog, pos: list, cfg: Config, rounds: int):
    rev = Prog(prog.logicals, prog.lindex, prog.ops[::-1], prog.gates[::-1], prog.m, prog.d0,
               [[] for _ in prog.gates], [])
    cands = []
    cur = list(pos)
    for _ in range(rounds):
        _, fin, _ = route(dev, prog, cur, cfg)
        _, back, _ = route(dev, rev, fin, cfg)
        cur = back
        cands.append(list(cur))
    return cands


def _pool_add(pool, score, pos, cap):
    if pool is None:
        return
    key = tuple(pos)
    if key in pool:
        return
    if len(pool) >= cap:
        worst = max(pool.values())
        if score >= worst:
            return
        for k2, v in list(pool.items()):
            if v == worst:
                del pool[k2]
                break
    pool[key] = score


def anneal_restarts(dev, prog, pos0, cfg: Config, deadline, rng, fresh, pool=None, evalf=None):
    if not cfg.sa_cycle:
        return anneal(dev, prog, pos0, cfg, deadline, rng, pool, evalf)
    total_cap = cfg.max_iters if cfg.max_iters is not None else 1 << 62
    ev = evalf or (lambda p: route(dev, prog, p, cfg)[0])
    best, best_pos = ev(pos0), list(pos0)
    used, cyc, stall = 0, 0, 0
    lb = 0.5 * prog.d0
    trig = cfg.pilot_trigger if (cfg.pilot and evalf is None) else 0
    while used < total_cap and time.perf_counter() < deadline and best > lb:
        if trig and stall >= trig:
            break
        n_it = min(cfg.sa_cycle, total_cap - used)
        if cfg.restart_mode == "mix" and cyc % 2 == 1:
            start = fresh(rng)
        else:
            start = best_pos
        b, p, it = anneal(dev, prog, start, replace(
            cfg, max_iters=n_it), deadline, rng, pool, evalf)
        used += max(it, 1)
        cyc += 1
        if b < best - 1e-12:
            best, best_pos = b, p
            stall = 0
        else:
            stall += 1
    return best, best_pos, used


def anneal(dev: Device, prog: Prog, pos0: list, cfg: Config, deadline: float, rng: random.Random, pool=None, evalf=None):
    n, m = dev.n, prog.m
    pos = list(pos0)
    occ = [-1] * n
    for l, p in enumerate(pos):
        occ[p] = l
    ev = evalf or (lambda p: route(dev, prog, p, cfg)[0])
    cur = ev(pos)
    best, best_pos = cur, list(pos)
    lb = 0.5 * prog.d0
    T = cfg.T0
    it = 0
    max_iters = cfg.max_iters if cfg.max_iters is not None else 1 << 62
    while it < max_iters and best > lb:
        if (it & 15) == 0 and time.perf_counter() > deadline:
            break
        it += 1
        q = rng.randrange(m)
        p = pos[q]
        if rng.random() < 0.5:
            t = rng.choice(dev.adj[p])
        else:
            t = rng.randrange(n - 1)
            if t >= p:
                t += 1
        o = occ[t]
        pos[q] = t
        occ[t] = q
        occ[p] = o
        if o >= 0:
            pos[o] = p
        new = ev(pos)
        dE = new - cur
        if dE <= 0 or rng.random() < math.exp(-dE / T):
            cur = new
            _pool_add(pool, new, pos, cfg.elite)
            if new < best:
                best, best_pos = new, list(pos)
        else:
            pos[q] = p
            occ[p] = q
            occ[t] = o
            if o >= 0:
                pos[o] = t
        T = max(cfg.Tmin, T * cfg.decay)
    return best, best_pos, it


def check(program, graph, placement, routed) -> bool:
    used = {q for op in program for q in op[1:]}
    if set(placement) != used:
        return False
    phys = list(placement.values())
    if len(set(phys)) != len(phys) or not set(phys) <= set(graph.nodes):
        return False
    p2l = {p: l for l, p in placement.items()}
    tr = []
    for op in routed:
        if op[0] == "SWAP":
            _, x, y = op
            if not graph.has_edge(x, y):
                return False
            lx, ly = p2l.get(x), p2l.get(y)
            p2l[x], p2l[y] = ly, lx
        elif op[0] == "2Q":
            _, x, y = op
            if not graph.has_edge(x, y) or p2l.get(x) is None or p2l.get(y) is None:
                return False
            tr.append(("2Q", p2l[x], p2l[y]))
        elif op[0] == "1Q":
            if p2l.get(op[1]) is None:
                return False
            tr.append(("1Q", p2l[op[1]]))
        else:
            return False
    return tr == [tuple(op) for op in program]


def score_of(routed) -> float:
    last: dict = {}
    depth = 0
    swaps = 0
    for op in routed:
        if op[0] == "1Q":
            continue
        if op[0] == "SWAP":
            swaps += 1
        layer = 1 + max(last.get(op[1], 0), last.get(op[2], 0))
        last[op[1]] = last[op[2]] = layer
        depth = max(depth, layer)
    return swaps + 0.5 * depth


def solve_ex(program, hardware_graph, cfg: Config = Config()):
    t_start = time.perf_counter()
    deadline_final = t_start + 0.9 * cfg.time_budget
    frac = cfg.pilot_frac if (cfg.pilot and not cfg.pilot_trigger) else 0.0
    deadline = t_start + 0.9 * cfg.time_budget * (1 - frac)
    sa_cfg = cfg
    if cfg.max_iters is not None and frac:
        sa_cfg = replace(cfg, max_iters=int(cfg.max_iters * (1 - frac)))
    info = {}
    program = [tuple(op) for op in program]
    if not program:
        return {}, [], {"method": "empty"}
    dev = device_for(hardware_graph)
    prog = prep_program(program)
    rng = random.Random(cfg.seed)
    lb = 0.5 * prog.d0

    pos = None
    emb = find_embedding(dev, prog, cfg.embed_budget)
    use_pilot = 0
    if emb is not None:
        pos, info["method"] = emb, "embed"
        best = route(dev, prog, pos, cfg)[0]
    else:
        starts = sorted(
            range(dev.n), key=lambda p: (-dev.deg[p], sum(dev.D[p])))[:4]
        cands = [greedy_placement(dev, prog, rng, s) for s in starts]
        if cfg.reverse_rounds:
            for c in list(cands):
                cands.extend(reverse_refine(
                    dev, prog, c, cfg, cfg.reverse_rounds))
        scored = sorted(
            (route(dev, prog, c, cfg)[0], i, c) for i, c in enumerate(cands))
        best, _, pos = scored[0]
        info["init_score"] = best
        if best > lb:
            def fresh(r):
                s = r.randrange(dev.n)
                p = greedy_placement(dev, prog, r, s)
                return p
            pool = {} if cfg.pilot else None
            _pool_add(pool, best, pos, cfg.elite)
            b2, p2, iters = anneal_restarts(
                dev, prog, pos, sa_cfg, deadline, rng, fresh, pool)
            info["sa_iters"] = iters
            if b2 < best:
                best, pos = b2, p2
            if cfg.reverse_rounds:
                for c in reverse_refine(dev, prog, pos, cfg, cfg.reverse_rounds):
                    s = route(dev, prog, c, cfg)[0]
                    if s < best:
                        best, pos = s, c
        info["method"] = "sa"
        use_pilot = 0
        if cfg.pilot and best > lb:
            if cfg.pilot_trigger:
                cap = (cfg.max_iters - info.get("sa_iters", 0)
                       ) if cfg.max_iters is not None else INF
            else:
                cap = cfg.max_iters * frac if cfg.max_iters is not None else INF
            cnt = [0.0]
            _pool_add(pool, best, pos, cfg.elite + 1)
            elites = sorted(pool.items(), key=lambda kv: kv[1])
            base_best = best
            n_ev = 0
            for key, _s in elites:
                if cnt[0] >= cap or time.perf_counter() > deadline_final:
                    break
                n_ev += 1
                if cfg.pilot_mode == 2:
                    s = route_swap_pilot(dev, prog, list(
                        key), cfg, cfg.pilot_horizon, cnt=cnt)[0]
                else:
                    s = route(dev, prog, list(key), cfg,
                              pilot=cfg.pilot, cnt=cnt)[0]
                if s < best - 1e-12 or (s <= best + 1e-12 and not use_pilot):
                    if s <= best + 1e-12:
                        best, pos, use_pilot = s, list(key), cfg.pilot
            if cfg.r2_sa and cfg.pilot_mode == 2 and n_ev and cnt[0] < cap and best > lb \
                    and time.perf_counter() < deadline_final:
                per_eval = cnt[0] / n_ev
                rem = cap - cnt[0]
                r2cfg = replace(cfg, sa_cycle=cfg.r2_cycle, T0=cfg.r2_T0,
                                max_iters=(max(1, int(rem / max(per_eval, 1e-9))) if cap < INF else None))
                if not use_pilot:
                    best = route_swap_pilot(
                        dev, prog, pos, cfg, cfg.pilot_horizon)[0]
                    use_pilot = cfg.pilot

                def evr2(p): return route_swap_pilot(
                    dev, prog, p, cfg, cfg.pilot_horizon, cnt=cnt)[0]
                b3, p3, it3 = anneal_restarts(
                    dev, prog, pos, r2cfg, deadline_final, rng, fresh, None, evr2)
                info["r2_iters"] = it3
                if b3 < best - 1e-12:
                    best, pos = b3, p3
            info["pilot_units"] = round(cnt[0])
            info["pilot_gain"] = base_best - best

    if use_pilot and cfg.pilot_mode == 2:
        score, _, routed_idx = route_swap_pilot(
            dev, prog, pos, cfg, cfg.pilot_horizon, emit=True)
    else:
        score, _, routed_idx = route(
            dev, prog, pos, cfg, emit=True, pilot=use_pilot)
    N = dev.nodes
    placement = {prog.logicals[l]: N[p] for l, p in enumerate(pos)}
    routed = []
    for op in routed_idx:
        if op[0] == "1Q":
            routed.append(("1Q", N[op[1]]))
        else:
            routed.append((op[0], N[op[1]], N[op[2]]))
    if not check(program, hardware_graph, placement, routed) or abs(score_of(routed) - score) > 1e-9:
        info["method"] = "fallback"
        placement, routed = _fallback(program, hardware_graph)
    info["score"] = score_of(routed)
    info["lb"] = lb
    info["time"] = time.perf_counter() - t_start
    return placement, routed, info


def _fallback(program, graph):
    logicals = sorted({q for op in program for q in op[1:]})
    nodes = sorted(graph.nodes)
    placement = {l: nodes[i] for i, l in enumerate(logicals)}
    cur = dict(placement)
    p2l = {p: l for l, p in cur.items()}
    out = []
    for op in program:
        if op[0] == "1Q":
            out.append(("1Q", cur[op[1]]))
            continue
        a, b = cur[op[1]], cur[op[2]]
        if not graph.has_edge(a, b):
            path = nx.shortest_path(graph, a, b)
            for x, y in zip(path[:-2], path[1:-1]):
                lx, ly = p2l.get(x), p2l.get(y)
                out.append(("SWAP", x, y))
                p2l[x], p2l[y] = ly, lx
                if lx is not None:
                    cur[lx] = y
                if ly is not None:
                    cur[ly] = x
        out.append(("2Q", cur[op[1]], cur[op[2]]))
    return placement, out


def solve(program, hardware_graph):
    placement, routed, _ = solve_ex(program, hardware_graph, Config())
    return placement, routed
