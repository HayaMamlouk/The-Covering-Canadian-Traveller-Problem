# experiments.py  – benchmark driver with robust per‑run time‑outs
# ------------------------------------------------------------------
import argparse, csv, math, random, time, multiprocessing as mp
from typing import List, Set, Tuple, Optional

import networkx as nx
from tqdm import tqdm

from cyclic_routing import cyclic_routing
from cnn_routing import cnn_routing

# ------------------------------------------------------------------
#  Basic helpers (unchanged)
# ------------------------------------------------------------------
def complete_euclidean(pts):
    G = nx.complete_graph(len(pts))
    for i, (x1, y1) in enumerate(pts):
        for j in range(i + 1, len(pts)):
            x2, y2 = pts[j]
            G[i][j]["weight"] = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return G


def christofides_lb(G, origin=0):
    from cnn_routing import _christofides_tour
    tour = _christofides_tour(G, origin)
    return sum(G[u][v]["weight"] for u, v in zip(tour, tour[1:] + [tour[0]]))


def walk_length(G, walk):
    total, prev = 0.0, walk[0]
    for cur in walk[1:]:
        if cur != prev:
            total += G[prev][cur]["weight"]
            prev = cur
    return total


def sample_blocked_edges_connected(G: nx.Graph, k: int,
                                   rnd: random.Random) -> Set[Tuple[int, int]]:
    max_safe = G.number_of_edges() - (G.number_of_nodes() - 1)
    k = min(k, max_safe)

    edges = list(G.edges())
    rnd.shuffle(edges)
    blocked, G_tmp = set(), G.copy()

    while len(blocked) < k and edges:
        u, v = edges.pop()
        G_tmp.remove_edge(u, v)
        if nx.is_connected(G_tmp):
            blocked.add((u, v))
        else:                          # keep connectivity
            G_tmp.add_edge(u, v, weight=G[u][v]["weight"])
    if len(blocked) < k:
        raise ValueError("could not find enough non‑bridge edges to block")
    return blocked

# ------------------------------------------------------------------
#  Graph families (unchanged)
# ------------------------------------------------------------------
_SMALL_GRAPHS = {
    3: [(0, 0), (1, 0), (0.5, math.sqrt(3) / 2)],
    4: [(0, 0), (1, 0), (1, 1), (0, 1)],
    6: [(math.cos(t), math.sin(t)) for t in
        [0, math.pi/3, 2*math.pi/3, math.pi,
         4*math.pi/3, 5*math.pi/3]],
    8: [(math.cos(t), math.sin(t)) for t in
        [i*math.pi/4 for i in range(8)]],
}
def family_A(n, s): return complete_euclidean(_SMALL_GRAPHS[n])

def family_B(n, s):
    r = random.Random(s)
    return complete_euclidean([(r.random(), r.random()) for _ in range(n)])

def _gaussian_clusters(n, s, k=4, sigma=0.03):
    r = random.Random(s)
    centres = [(r.random(), r.random()) for _ in range(k)]
    per = n // k
    pts = [(r.gauss(cx, sigma), r.gauss(cy, sigma))
           for cx, cy in centres for _ in range(per)]
    pts += [(r.random(), r.random()) for _ in range(n-len(pts))]
    return pts
def family_C(n, s): return complete_euclidean(_gaussian_clusters(n, s))

def _grid_with_highways(m):
    pts = [(i, j) for i in range(m) for j in range(m)]
    G = complete_euclidean(pts)
    for i in range(m):
        u, v = i*m+i, i*m+(m-1-i)
        G[u][v]["weight"] *= 0.3
    return G
def family_D(n, s): return _grid_with_highways(int(round(math.sqrt(n))))

def family_E(n, s):
    G = family_B(n, s)
    from cnn_routing import _christofides_tour
    tour = _christofides_tour(G, 0)
    k_adv = int(math.ceil(0.8*n))
    start = s % n
    blocked = {tuple(sorted((tour[(start+i)%n], tour[(start+i+1)%n])))
               for i in range(k_adv)}
    return G, blocked

FAMILIES = {"A": family_A, "B": family_B, "C": family_C,
            "D": family_D, "E": family_E}

# ------------------------------------------------------------------
#  Timeout‑safe wrapper for CR
# ------------------------------------------------------------------
def _cr_worker(G, blocked, q):
    walk, _ = cyclic_routing(G, 0, blocked)
    q.put(walk)

def safe_route(algo: str, G: nx.Graph, blocked, timeout: int):
    """Run <algo>; kill it after *timeout* seconds if needed."""
    if algo == "CNN":                       # fast → run in‑process
        t0 = time.perf_counter()
        walk, _ = cnn_routing(G, 0, blocked)
        ms = int((time.perf_counter()-t0)*1000)
        return "OK", walk, ms

    # --- potentially slow CR: isolate in its own process -------------
    q = mp.Queue()
    p = mp.Process(target=_cr_worker, args=(G, blocked, q))
    t0 = time.perf_counter()
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return "TIMEOUT", None, timeout*1000
    walk = q.get()
    ms = int((time.perf_counter()-t0)*1000)
    return "OK", walk, ms

# ------------------------------------------------------------------
#  Main benchmark loop
# ------------------------------------------------------------------
def k_vals(n, explicit):
    return explicit if explicit is not None else [0, int(n**0.5), int(0.3*n)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["B"])
    ap.add_argument("--sizes",    nargs="+", type=int,
                    default=[20, 40, 80, 160])
    ap.add_argument("--seeds",    type=int, default=30)
    ap.add_argument("--kvals",    nargs="+", type=int)
    ap.add_argument("--algos",    nargs="+", choices=["CR", "CNN"],
                    default=["CR", "CNN"])
    ap.add_argument("--timeout",  type=int, default=30,
                    help="wall‑clock limit (seconds) for each CR instance")
    ap.add_argument("--out",      default="results.csv")
    args = ap.parse_args()

    total = (len(args.families) *
            len(args.algos) *
            sum(args.seeds * len(k_vals(n, args.kvals))
                for n in args.sizes))
    rows, bar = [], tqdm(total=total, unit="run", colour="green")

    for fam_key in args.families:
        fam_fun = FAMILIES[fam_key]
        for n in args.sizes:
            for seed in range(args.seeds):
                g_or_pair = fam_fun(n, seed)
                if isinstance(g_or_pair, tuple):
                    G, pre_blocked = g_or_pair
                else:
                    G, pre_blocked = g_or_pair, None
                offline = christofides_lb(G)

                for k in k_vals(n, args.kvals):
                    blocked = (pre_blocked if pre_blocked is not None else
                               sample_blocked_edges_connected(
                                   G, k, random.Random(seed+997*k)))

                    for algo in args.algos:
                        status, walk, ms = safe_route(
                            algo, G, blocked, args.timeout)

                        if status == "OK":
                            tlen = walk_length(G, walk)
                            rows.append([fam_key,n,k,seed,algo,
                                         tlen,offline,tlen/offline,ms,status])
                        else:
                            rows.append([fam_key,n,k,seed,algo,
                                         "", "", "", ms, status])
                        bar.update(1)
    bar.close()

    with open(args.out,"w",newline="") as fh:
        csv.writer(fh).writerows(
            [["family","n","k","seed","algo",
              "tour_len","offline_opt_lb","competitive_ratio",
              "time_ms","status"]] + rows)

if __name__ == "__main__":
    mp.freeze_support()      # so Windows can spawn safely
    main()
