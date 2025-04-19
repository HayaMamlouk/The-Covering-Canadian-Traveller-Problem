"""
CNN *compress‑and‑explore* algorithm
===================================

The algorithm consists of two phases:

1. **Short‑Cut phase** – follow the 1.5‑approximate Christofides TSP tour as
   far as possible, *skipping* vertices whose incident edge is discovered
   blocked.  Every time the traveller is at a vertex, it learns the status of
   all incident edges.  Blocked edges are collected into :math:`E_b`.
   The walk produced in this phase is ``P1``;  the set of still‑unvisited
   vertices *plus* the origin is :math:`U_s`.

2. **Compression & Exploration phase** – build a *compressed* auxiliary graph
   :math:`G'` consisting only of the vertices in :math:`U_s` and, for each pair
   `(x, y)`, an edge whose weight equals the length of the *shortest path* in
   the partially known graph (all blocked edges removed) that connects `x` and
   `y` using only *known‑to‑exist* edges.  Run the classic *Nearest‑Neighbor*
   greedy exploration algorithm on :math:`G'` (second walk ``P2``) and finally
   **expand** every compressed edge back into its real path.

The final tour is ``P = P1  ∘  P2``.

--------------------------------------------------------------------------
API
--------------------------------------------------------------------------

``cnn_routing(G, origin, blocked_edges)`` → ``(route, length)``

* ``route``  – list of vertices starting and ending at ``origin``.
* ``length`` – total metric length according to edge weights in ``G``.

The code requires **networkx ≥ 2.8**.  It automatically handles both the older
and newer calling conventions of ``christofides`` inside NetworkX ≥ 3.2.
"""
from __future__ import annotations

import heapq
from itertools import tee
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx

# ---------------------------------------------------------------------
# Utilities – reused from cyclic_routing.py (kept self‑contained here)
# ---------------------------------------------------------------------

def _edge_key(u, v):
    """Return a canonical undirected 2‑tuple suitable for set/dict keys."""
    return (u, v) if u <= v else (v, u)


def _pairwise(it):
    """s -> (s0,s1), (s1,s2), …"""
    a, b = tee(it)
    next(b, None)
    return zip(a, b)

# ---------------------------------------------------------------------
# Christofides 1.5‑approximation tour (same patch as in cyclic_routing.py)
# ---------------------------------------------------------------------

def _greedy_fallback(G: nx.Graph, origin: int) -> List[int]:
    """Very small fallback if christofides() unavailable."""
    unvisited = set(G.nodes())
    unvisited.remove(origin)
    tour = [origin]
    cur = origin
    while unvisited:
        nxt = min(unvisited, key=lambda v: G[cur][v]["weight"])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour


def _christofides_tour(G: nx.Graph, origin: int) -> List[int]:
    if not hasattr(nx.algorithms.approximation, "christofides"):
        return _greedy_fallback(G, origin)

    from networkx.algorithms.approximation.traveling_salesman import (
        traveling_salesman_problem as _tsp,
    )

    try:
        tour = _tsp(
            G,
            cycle=True,
            weight="weight",
            method=nx.algorithms.approximation.christofides,
        )
    except TypeError:  # older NetworkX expected 'start'
        tour = _tsp(
            G,
            cycle=True,
            weight="weight",
            method=nx.algorithms.approximation.christofides,
            start=origin,
        )

    # Rotate so that tour starts at *origin* and drop duplicate closing vertex.
    if tour[0] != origin:
        k = tour.index(origin)
        tour = tour[k:-1] + tour[:k]
    else:
        tour = tour[:-1]
    return tour

# ---------------------------------------------------------------------
# 1. Short‑Cut phase (lines 1‑20 of the pseudocode)                    |
# ---------------------------------------------------------------------

def _shortcut_phase(
    G: nx.Graph,
    P: Sequence[int],
    blocked_edges: Set[Tuple[int, int]],
):
    """Perform the ShortCut procedure.

    Returns
    -------
    G_star : nx.Graph  – graph with *all discovered blocked edges removed*.
    Us      : set      – unvisited vertices *plus* the origin.
    P1      : list     – realised walk of the traveller in this phase.
    """
    n = len(P)
    V = set(G.nodes())
    s = P[0]

    i, j = 0, 1  # indices in *P*
    Us = {s}
    Eb: Set[Tuple[int, int]] = set()
    P_prime: List[int] = [s]

    def is_blocked(u, v):
        return _edge_key(u, v) in blocked_edges

    while j < n:
        v_i, v_j = P[i], P[j]

        # (5) add newly discovered blocked edges adjacent to v_i
        for x in V - {v_i}:
            if is_blocked(v_i, x):
                Eb.add(_edge_key(v_i, x))

        # (6‑11)
        if not is_blocked(v_i, v_j):
            P_prime.append(v_j)
            i = j  # advance – we really moved to v_j
        else:
            Us.add(v_j)  # mark v_j unvisited for later phase
            # we stay on v_i; i unchanged
        j += 1

    # After the loop, we are still at P[i]. Try to return to s.
    v_i = P[i]
    if is_blocked(v_i, s):
        # (15‑16) return following P' backwards
        backtrack = list(reversed(P_prime[:-1]))  # skip current v_i duplicate
        P_prime.extend(backtrack)
    else:
        # (18) directly append s
        P_prime.append(s)

    # Build G* (blocked edges removed)
    G_star = G.copy()
    for (u, v) in Eb:
        if G_star.has_edge(u, v):
            G_star.remove_edge(u, v)

    return G_star, Us, P_prime

# ---------------------------------------------------------------------
# 2. Compression phase (Function Compress)                             |
# ---------------------------------------------------------------------

def _compress_phase(
    G_star: nx.Graph,
    Us: Set[int],
):
    """Return the compressed multigraph G' and a path‑lookup table.

    The lookup maps an **unordered** pair {u, v} (as a frozenset) to the actual
    shortest path (list of vertices) inside *G_star*.
    """
    G_prime = nx.Graph()
    G_prime.add_nodes_from(Us)

    path_lookup: Dict[frozenset[int], List[int]] = {}

    # Pre‑compute all‑pairs shortest paths only between Us vertices.
    Us_list = list(Us)
    for i in range(len(Us_list)):
        for j in range(i + 1, len(Us_list)):
            u, v = Us_list[i], Us_list[j]
            try:
                length, path = nx.single_source_dijkstra(G_star, u, v, weight="weight")
            except nx.NetworkXNoPath:
                continue  # keep graph disconnected; exploration will handle
            G_prime.add_edge(u, v, weight=length)
            path_lookup[frozenset((u, v))] = path

    return G_prime, path_lookup

# ---------------------------------------------------------------------
# 3. Nearest‑Neighbor exploration on G'                                |
# ---------------------------------------------------------------------

def _nearest_neighbor_tour(G: nx.Graph, origin: int) -> List[int]:
    """Return an NN tour *starting and ending* at ``origin``."""
    unvisited = set(G.nodes())
    unvisited.remove(origin)
    tour = [origin]
    cur = origin
    while unvisited:
        nxt = min(unvisited, key=lambda v: G[cur][v]["weight"])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    tour.append(origin)
    return tour


def _expand_compressed_tour(
    compressed_tour: Sequence[int],
    path_lookup: Dict[frozenset[int], List[int]],
):
    """Expand every compressed edge into its real path inside the original graph."""
    expanded: List[int] = [compressed_tour[0]]
    for u, v in _pairwise(compressed_tour):
        key = frozenset((u, v))
        real_path = path_lookup.get(key, [u, v])  # fallback to direct edge
        if real_path[0] == expanded[-1]:
            expanded.extend(real_path[1:])
        else:  # path stored in opposite direction
            expanded.extend(list(reversed(real_path))[1:])
    return expanded

# ---------------------------------------------------------------------
# 4. Public wrapper – CNN routing                                      |
# ---------------------------------------------------------------------

def cnn_routing(
    G: nx.Graph,
    origin: int,
    blocked_edges: Set[Tuple[int, int]] | List[Tuple[int, int]] = (),
):
    """Run the *Christofides' Nearest‑Neighbor* algorithm.

    Parameters
    ----------
    G : ``networkx.Graph`` – **complete** metric graph (triangle inequality).
    origin : starting vertex ``s``.
    blocked_edges : collection of unordered vertex pairs that are permanently
                    blocked (concealed from the algorithm until first probe).

    Returns
    -------
    route  – list of vertices (walk) starting and ending at *origin*.
    length – total distance travelled.
    """
    # Normalise blocked‑edge set
    blocked_edges = {_edge_key(u, v) for (u, v) in blocked_edges if u != v}

    # 1. Christofides tour
    P = _christofides_tour(G, origin)

    # 2. Short‑Cut phase
    G_star, Us, P1 = _shortcut_phase(G, P, blocked_edges)

    # 3. Compression phase
    G_prime, path_lookup = _compress_phase(G_star, Us)

    # 4. Nearest‑Neighbor exploration on compressed graph
    P2_compressed = _nearest_neighbor_tour(G_prime, origin)
    P2 = _expand_compressed_tour(P2_compressed, path_lookup)

    # 5. Concatenate paths (skip duplicate origin in the middle)
    final_route = P1 + P2[1:]

    # 6. Total length (skip consecutive duplicates, use *G* as metric authority)
    total_len = 0.0
    prev = final_route[0]
    for cur in final_route[1:]:
        if cur == prev:
            continue  # ignore 0‑length self‑loops that may appear after expansion
        try:
            total_len += G[prev][cur]["weight"]
        except KeyError as err:
            raise KeyError(
                f"Edge ({prev}, {cur}) not present in the original graph. "
                "Ensure the input G is complete (metric) or amend the code to "
                "compute path lengths via shortest‑path instead."
            ) from err
        prev = cur

    return final_route, total_len

# ---------------------------------------------------------------------
# Quick self‑test when invoked directly                                 
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import random, math

    random.seed(42)
    n = 15
    G = nx.complete_graph(n)
    coords = {v: (random.random(), random.random()) for v in G}
    for u, v in G.edges:
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        G[u][v]["weight"] = math.hypot(x1 - x2, y1 - y2)

    blocked = {(0, 2), (1, 3), (4, 7), (5, 6), (8, 9), (10, 11)}

    route, length = cnn_routing(G, origin=0, blocked_edges=blocked)

    print("CNN route (|V| =", n, "):")
    print(route)
    print("Total length:", length)
