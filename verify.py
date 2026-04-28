#!/usr/bin/env python3
"""Certification of the 3x3 matmul, rank 23, 56-addition algorithm.

Reconstructs U, V, W from the inlined CSE program (no npz, no JSON), then:
  (1) checks Brent's identity over Z, GF(2), GF(3)  -> matmul correctness, rank 23
  (2) audits addition counts (U=13, V=13, W=30, total=56)
  (3) checks input-side optimality lower bound: U and V each need >= 13 additions

Conventions: A, B, C are flattened row-major: A[i,l] -> 3*i+l, etc.
"""

SIDES = {
    'U': {
        'base_dim': 9,
        'inter': [(6,-1,7),(4,-1,9),(1,-1,10),(11,-1,6),(12,-1,8),(1,-1,13),
                  (0,-1,11),(13,1,2),(15,-1,3),(17,1,5),(18,-1,2),(0,-1,1),(20,-1,18)],
        'final': [[(14,1)],[(21,1)],[(5,1)],[(16,1)],[(13,1)],[(4,1)],[(8,1)],
                  [(10,-1)],[(4,1)],[(11,1)],[(12,-1)],[(8,1)],[(15,1)],[(2,1)],
                  [(1,1)],[(6,1)],[(17,1)],[(0,1)],[(5,1)],[(3,1)],[(19,-1)],
                  [(9,1)],[(18,1)]],
    },
    'V': {
        'base_dim': 9,
        'inter': [(2,1,5),(3,-1,5),(1,1,4),(8,-1,9),(0,-1,11),(2,-1,13),
                  (4,-1,14),(7,-1,15),(2,-1,16),(8,1,16),(10,-1,14),(8,1,19),(6,-1,20)],
        'final': [[(19,-1)],[(15,-1)],[(18,1)],[(8,1)],[(20,-1)],[(5,1)],[(7,1)],
                  [(14,1)],[(3,1)],[(13,1)],[(12,-1)],[(21,1)],[(2,1)],[(6,1)],
                  [(3,1)],[(11,1)],[(17,1)],[(0,1)],[(6,1)],[(0,1)],[(7,1)],
                  [(4,1)],[(16,-1)]],
    },
    'W': {
        'base_dim': 23,
        'inter': [(4,1,9),(12,1,22),(14,1,23),(0,1,7),(1,-1,7),(10,1,25),(16,-1,24)],
        'final': [
            [(13,1),(14,1),(17,1)],
            [(9,-1),(17,1),(20,1),(24,-1),(27,1)],
            [(3,1),(12,1),(25,1),(26,1)],
            [(8,1),(18,1),(19,1)],
            [(19,1),(21,1),(27,1),(29,1)],
            [(2,1),(5,1),(29,-1)],
            [(8,-1),(11,1),(15,1),(28,1)],
            [(6,1),(15,1),(21,-1)],
            [(5,-1),(26,1),(28,1)],
        ],
    },
}

TPERM = [0,3,6,1,4,7,2,5,8]


def expand(side):
    n = side['base_dim']
    vs = [tuple(1 if i == j else 0 for j in range(n)) for i in range(n)]
    for a, s, b in side['inter']:
        va, vb = vs[a], vs[b]
        vs.append(tuple(va[i] + s*vb[i] for i in range(n)))
    out = []
    for f in side['final']:
        acc = [0]*n
        for idx, c in f:
            v = vs[idx]
            for i in range(n):
                acc[i] += c * v[i]
        out.append(tuple(acc))
    return out


def cost(side):
    extra = sum(max(sum(abs(c) for _, c in f) - 1, 0) for f in side['final'])
    return len(side['inter']) + extra


U = expand(SIDES['U'])             # 23 rows of length 9   (column k <-> A[k//3, k%3])
V = expand(SIDES['V'])             # 23 rows of length 9   (column k <-> B[k//3, k%3])
W_cols = expand(SIDES['W'])        # 9 columns of length 23
RANK = 23
W = [tuple(W_cols[c][r] for c in range(9)) for r in range(RANK)]   # row r of products


def brent(mod=None):
    bad = 0
    for i in range(3):
        for l in range(3):
            for lp in range(3):
                for j in range(3):
                    for ip in range(3):
                        for jp in range(3):
                            a, b, c = 3*i+l, 3*lp+j, 3*ip+jp
                            s = sum(U[r][a]*V[r][b]*W[r][c] for r in range(RANK))
                            d = s - (1 if (i == ip and j == jp and l == lp) else 0)
                            if mod is not None:
                                d %= mod
                            if d:
                                bad += 1
    return bad

D = 9
ZERO = (0,) * D
BASIS = [tuple(1 if i == j else 0 for j in range(D)) for i in range(D)]


def canon(v):
    t = tuple(v); return min(t, tuple(-x for x in t))


ICAN = {canon(e) for e in BASIS}


def add(a, b, sa=1, sb=1):
    return tuple(sa*a[i] + sb*b[i] for i in range(D))


def targets(rows):
    out = []
    for r in rows:
        c = canon(r)
        if c not in ICAN and c not in out:
            out.append(c)
    return out


def pure_deps(T):
    forms = BASIS + T
    masks = [0]*len(BASIS) + [1 << i for i in range(len(T))]
    deps = []
    for ti, t in enumerate(T):
        opts = set()
        for a in range(len(forms)):
            if masks[a] & (1 << ti): continue
            for b in range(a):
                if masks[b] & (1 << ti): continue
                for sa in (1, -1):
                    for sb in (1, -1):
                        g = add(forms[a], forms[b], sa, sb)
                        if g != ZERO and canon(g) == t:
                            opts.add(masks[a] | masks[b])
        deps.append(sorted(opts))
    return deps


def reachable(T, target_deps, aux_deps=()):
    N = len(T) + len(aux_deps)
    full = (1 << len(T)) - 1
    deps = [list(x) for x in target_deps] + [list(x) for x in aux_deps]
    seen = {0}; stack = [0]
    while stack:
        m = stack.pop()
        if (m & full) == full:
            return True
        for node in range(N):
            if m >> node & 1: continue
            if any((req & ~m) == 0 for req in deps[node]):
                nm = m | (1 << node)
                if nm not in seen:
                    seen.add(nm); stack.append(nm)
    return False


def aux1_candidates(T):
    forms = BASIS + T
    masks = [0]*len(BASIS) + [1 << i for i in range(len(T))]
    Tset = set(T); out = {}
    for a in range(len(forms)):
        for b in range(a):
            for sa in (1, -1):
                for sb in (1, -1):
                    c = canon(add(forms[a], forms[b], sa, sb))
                    if c == ZERO or c in ICAN or c in Tset: continue
                    out.setdefault(c, set()).add(masks[a] | masks[b])
    return out


def deps_with_aux(T, aux, bit):
    forms = BASIS + T
    masks = [0]*len(BASIS) + [1 << i for i in range(len(T))]
    out = [set() for _ in T]
    for ix, x in enumerate(forms):
        xm = masks[ix]
        for ti, t in enumerate(T):
            if xm & (1 << ti): continue
            for sa in (1, -1):
                for sx in (1, -1):
                    if canon(add(aux, x, sa, sx)) == t:
                        out[ti].add(bit | xm)
    return out


def optimality():
    invT = [0]*9
    for j, k in enumerate(TPERM):
        invT[k] = j
    oldW_rows = [tuple(row[invT[k]] for k in range(9)) for row in V]

    TU = targets(U); depsU = pure_deps(TU)
    TV = targets(oldW_rows); depsV = pure_deps(TV)
    U_pure = reachable(TU, depsU)
    V_pure = reachable(TV, depsV)
    V_q1 = False
    if not V_pure:
        bit = 1 << len(TV)
        for av, ad in aux1_candidates(TV).items():
            dax = deps_with_aux(TV, av, bit)
            deps = [set(depsV[i]) | dax[i] for i in range(len(TV))]
            if reachable(TV, deps, [ad]):
                V_q1 = True; break
    return {
        'U_targets': len(TU), 'U_pure_at_12_possible': U_pure,
        'U_lb': 13 if not U_pure else len(TU),
        'V_targets': len(TV), 'V_pure_at_11_possible': V_pure,
        'V_aux1_at_12_possible': V_q1,
        'V_lb': 13 if (not V_pure and not V_q1) else None,
    }


if __name__ == '__main__':
    cU, cV, cW = cost(SIDES['U']), cost(SIDES['V']), cost(SIDES['W'])
    print(f"rank                     : {RANK}")
    print(f"additions  U / V / W     : {cU} / {cV} / {cW}   (total {cU+cV+cW})")
    for mod in (None, 2, 3):
        bad = brent(mod)
        label = 'Z' if mod is None else f'GF({mod})'
        print(f"Brent over {label:6s}        : {'OK' if bad == 0 else f'FAIL ({bad})'}")
    opt = optimality()
    print(f"U distinct targets       : {opt['U_targets']}   "
          f"pure-chain at 12 possible: {opt['U_pure_at_12_possible']}   "
          f"lb = {opt['U_lb']}")
    print(f"V distinct targets       : {opt['V_targets']}   "
          f"pure-chain at 11 possible: {opt['V_pure_at_11_possible']}   "
          f"+1aux at 12 possible: {opt['V_aux1_at_12_possible']}   "
          f"lb = {opt['V_lb']}")
