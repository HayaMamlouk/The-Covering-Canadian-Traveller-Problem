# MAMLOUK Haya [21107689]
# OZGENC Doruk [21113927]
# ------------------------------------------------------------------

import argparse, csv, math, multiprocessing as mp, queue, random, time
from typing import List 

import networkx as nx
from tqdm import tqdm

from cyclic_routing import cyclic_routing
from cnn_routing import cnn_routing


# ------------------------------------------------------------------
#  Basic helpers 
# ------------------------------------------------------------------
def complete_euclidean(pts):
    """Build a complete graph with Euclidean distances."""
    # pts = [(x, y), ...]
    G = nx.complete_graph(len(pts))
    for i, (x1, y1) in enumerate(pts):
        for j in range(i + 1, len(pts)):
            x2, y2 = pts[j]
            G[i][j]["weight"] = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return G


def christofides_lb(G, origin: int = 0):
    """Compute the Christofides lower bound for G."""
    from cnn_routing import get_christofides_tour

    tour = get_christofides_tour(G, origin)
    return sum(G[u][v]["weight"] for u, v in zip(tour, tour[1:] + [tour[0]]))


def walk_length(G, walk):
    """Compute the length of a walk in G."""
    total, prev = 0.0, walk[0]
    for cur in walk[1:]:
        if cur != prev and G.has_edge(prev, cur):
            total += G[prev][cur]["weight"]
            prev = cur
    return total


def sample_blocked_edges_connected(G, k, rnd: random.Random) :
    """Choose k non‑bridge edges so that G \ blocked stays connected."""
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
        else:  # keep connectivity
            G_tmp.add_edge(u, v, weight=G[u][v]["weight"])
    return blocked


# ------------------------------------------------------------------
#  Graph‑families (B, C, D, E, H)
# ------------------------------------------------------------------
def family_B(n, s):
    """Generate n random points in the unit square."""
    r = random.Random(s)
    return complete_euclidean([(r.random(), r.random()) for _ in range(n)])

# --- CLUSTERS -----------------------------------------------------

def _gaussian_clusters(n, s, k=4, sigma=0.03):
    """Generate n points in k clusters, each with a Gaussian distribution."""
    r = random.Random(s)
    centres = [(r.random(), r.random()) for _ in range(k)]
    per = n // k
    pts = [(r.gauss(cx, sigma), r.gauss(cy, sigma))
           for cx, cy in centres for _ in range(per)]
    pts += [(r.random(), r.random()) for _ in range(n - len(pts))]
    return pts


def family_C(n, s):
    """Generates compact clusters seperated by bridges."""
    return complete_euclidean(_gaussian_clusters(n, s))

# --- GRID + DIAGONALS -----------------------------------------

def _grid_with_highways(n, seed):
    """
    Build an m*m grid. Make the two main diagonals 30 % cheaper.
    """
    m = math.ceil(math.sqrt(n))
    pts = [(i, j) for i in range(m) for j in range(m)]
    G = complete_euclidean(pts)
    # cheap highways on the full m*m grid
    for i in range(m):
        u, v = i * m + i, i * m + (m - 1 - i)
        if u != v and G.has_edge(u, v):
            G[u][v]["weight"] *= 0.3
    # keep only the first n nodes (deterministic)
    if m * m > n:
        keep = list(range(n))
        G = G.subgraph(keep).copy()
    return G


def family_D(n, s):
    return _grid_with_highways(n, s)

# --- ADVERSARIAL CHRISTOFIDES ------------------------------------

def family_E(n, s):
    """Blocks around 80 % of the edges in a Christofides tour."""
    G = family_B(n, s)
    from cnn_routing import get_christofides_tour

    tour = get_christofides_tour(G, 0)
    k_adv = int(math.ceil(0.8 * n))
    start = s % n
    blocked = {tuple(sorted((tour[(start + i) % n], tour[(start + i + 1) % n])))
               for i in range(k_adv)}
    return G, blocked

# --- HUB‑AND‑SPOKE / STAR --------------------------------------

def family_H(n: int, s: int) -> nx.Graph:
    """Star metric : node 0 is the hub (cost 1), all others are leaves (leaf-leaf = 1)."""
    if n < 3:
        raise ValueError("Hub‑and‑spoke needs n ≥ 3")

    G = nx.complete_graph(n)
    for i in range(1, n):
        G[0][i]["weight"] = 1.0  # branches
    for i in range(1, n):
        for j in range(i + 1, n):
            G[i][j]["weight"] = 2.0  # direct leaf‑to‑leaf (equals detour via hub)
    return G  

#---------------------------------------------------------------------  

FAMILIES = {"B": family_B, "C": family_C,
            "D": family_D, "E": family_E, "H": family_H}

# ------------------------------------------------------------------
#  Timeout‑safe wrapper for CR
# ------------------------------------------------------------------
def _cr_worker(G, blocked, q):
    try:
        walk, _ = cyclic_routing(G, 0, blocked)
        q.put(("OK", walk))
    except Exception as exc:           # any hard error → record & return
        q.put(("ERROR", str(exc)))


def safe_route(algo: str, G, blocked,timeout: int):
    """Run <algo>; kill it after *timeout* s if needed."""
    if algo == "CNN":                 # fast – run in‑process
        t0 = time.perf_counter()
        walk, _ = cnn_routing(G, 0, blocked)
        ms = int((time.perf_counter() - t0) * 1000)
        return "OK", walk, ms

    # --- potentially slow CR → isolate in its own process -------------
    q: mp.Queue = mp.Queue(maxsize=1)
    p = mp.Process(target=_cr_worker, args=(G, blocked, q))
    t0 = time.perf_counter()
    p.start()
    p.join(timeout)
    if p.is_alive():                  # timed‑out – kill it
        p.terminate()
        p.join()
        return "TIMEOUT", None, timeout * 1000

    try:
        status, payload = q.get_nowait()
    except queue.Empty:
        return "ERROR", None, int((time.perf_counter() - t0) * 1000)

    if status == "OK":
        walk = payload
        ms = int((time.perf_counter() - t0) * 1000)
        return "OK", walk, ms
    else:                             # crash inside CR
        ms = int((time.perf_counter() - t0) * 1000)
        return "ERROR", None, ms


# ------------------------------------------------------------------
#  Main benchmark loop
# ------------------------------------------------------------------
# def default_k_vals(n: int):
#     return [0, int(n ** 0.5), int(0.2 * n)]

def default_k_vals(n):
    return [0,
            1,
            int(0.05*n),
            int(0.1*n),
            int(0.2*n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+",
                    default=list(FAMILIES))  # A B C D E
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[20, 40, 80, 160])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--kvals", nargs="+", type=int)
    ap.add_argument("--algos", nargs="+", choices=["CR", "CNN"],
                    default=["CR", "CNN"])
    ap.add_argument("--timeout", type=int, default=10,
                    help="wall-clock limit (seconds) for EACH CR instance")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    k_values = (lambda n: args.kvals) if args.kvals else default_k_vals
    total = (
        len(args.families) *
        len(args.algos) *
        sum(args.seeds * len(k_values(n)) for n in args.sizes)
    )
    rows: List[List] = []
    bar = tqdm(total=total, unit="run", colour="green")

    try:
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

                    for k in k_values(n):
                        blocked = (pre_blocked if pre_blocked is not None else
                                   sample_blocked_edges_connected(
                                       G, k, random.Random(seed + 997 * k)))

                        for algo in args.algos:
                            status, walk, ms = safe_route(
                                algo, G, blocked, args.timeout)

                            if status == "OK":
                                tlen = walk_length(G, walk)
                                rows.append([fam_key, n, k, seed, algo,
                                             tlen, offline, tlen / offline,
                                             ms, status])
                            else:  # TIMEOUT or ERROR
                                rows.append([fam_key, n, k, seed, algo,
                                             "", "", "", ms, status])
                            bar.update(1)
    finally:
        bar.close()

    with open(args.out, "w", newline="") as fh:
        csv.writer(fh).writerows(
            [["family", "n", "k", "seed", "algo",
              "tour_len", "offline_opt_lb", "competitive_ratio",
              "time_ms", "status"]] + rows
        )


if __name__ == "__main__":
    mp.freeze_support()  # needed on Windows
    main()
