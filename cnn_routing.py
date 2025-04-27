# MAMLOUK Haya [21107689]
# OZGENC Doruk [21113927]

from __future__ import annotations
from itertools import tee, combinations
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem
from typing import Dict, List

# ---------------------------------------------------------------------
# Utilities 
# ---------------------------------------------------------------------
def _sorted_edge(u, v) :
    """Return a sorted tuple of the edge (u, v) to ensure undirectedness."""
    return tuple(sorted((u, v)))

def _pairwise(it):
    """s -> (s0,s1), (s1,s2), …"""
    a, b = tee(it)
    next(b, None)
    return zip(a, b)
# ----------------------------------------------------------------------
# Christofides Tour (approximation for TSP)
# ----------------------------------------------------------------------
def get_christofides_tour(G, start_node):
    """
    Compute a Christofides TSP tour on G, rotate to start at start_node,
    and drop the final return to that node to get a simple cycle ordering.
    """
    cycle = traveling_salesman_problem(G, cycle=True, weight='weight')
    if start_node in cycle:
        idx = cycle.index(start_node)
        cycle = cycle[idx:] + cycle[:idx]
    return cycle[:-1]  # remove duplicated endpoint

# ---------------------------------------------------------------------
# 1. Short‑Cut phase
# ---------------------------------------------------------------------

def _shortcut_phase(G, P, blocked_edges):
    """Perform the ShortCut procedure.
    Returns
    -------
    G_star : nx.Graph  - graph with all discovered blocked edges removed*.
    Us      : set      - unvisited vertices plus the origin.
    P1      : list     - realised walk of the traveller in this phase.
    """
    n = len(P)           # number of vertices in the tour
    V = set(G.nodes())   # all vertices in the graph
    s = P[0]             # starting vertex (origin)

    i, j = 0, 1  # indices of the current and next vertex in the tour
    Us = {s}     # unvisited vertices (initially only the origin)
    Eb = set() # discovered blocked edges
    P_prime = [s]         # realised walk of the traveller

    def is_blocked(u, v):
        return _sorted_edge(u, v) in blocked_edges

    while j < n:
        v_i, v_j = P[i], P[j]  # current and next vertex in the tour

        #  newly discovered blocked edges adjacent to v_i
        for x in V - {v_i}:
            if is_blocked(v_i, x):
                Eb.add(_sorted_edge(v_i, x))

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
        backtrack = list(reversed(P_prime[:-1]))  # skip current v_i duplicate
        P_prime.extend(backtrack)
    else:
        P_prime.append(s)

    # Build G* (blocked edges removed)
    G_star = G.copy()
    for (u, v) in Eb:
        if G_star.has_edge(u, v):
            G_star.remove_edge(u, v)

    return G_star, Us, P_prime

# ---------------------------------------------------------------------
# 2. Compression phase (Function Compress)    
# ---------------------------------------------------------------------
def _compress_phase(G_star, Us):
    """Return the compressed multigraph G' and a path lookup table.

    The lookup maps an unordered pair {u,v} (as a frozenset) to the actual
    shortest path (list of vertices) inside G_star.
    """
    #E′ is the subset of edges with unknown state, i.e., of {x,y} with x,y∈Us.
    E_prime = [(u, v) for u in Us for v in Us if u != v]

    # G' is the graphe with Us vertices and edges in E′.
    G_prime = nx.Graph()
    G_prime.add_nodes_from(Us)
    # for u, v in E_prime:
    #     w = G_star[u][v]["weight"] 
    #     G_prime.add_edge(u, v, weight=w)

    for u, v in combinations(Us, 2):          # each pair once
        if G_star.has_edge(u, v):
            w = G_star[u][v]["weight"]
            G_prime.add_edge(u, v, weight=w)     

    # H = (V, E\E')
    H = G_star.copy()
    H.remove_edges_from(E_prime)

    path_lookup: Dict[frozenset[int], List[int]] = {}
    
    Us_list = list(Us)

    for i, u in enumerate(Us_list):
        for v in Us_list[i + 1:]:
            # Compute the shortest path between u and v in H
            path = nx.shortest_path(H, source=u, target=v, weight="weight")
            length = sum(G_star[u][v]["weight"] for u, v in _pairwise(path))
            # Add the path to G'
            G_prime.add_edge(u, v, weight=length)
            path_lookup[frozenset((u, v))] = path
    return G_prime, path_lookup
            
# ---------------------------------------------------------------------
# 3. Nearest‑Neighbor exploration on G'     
# ---------------------------------------------------------------------

def _nearest_neighbor_tour(G, origin):
    """Return an NN tour starting and ending at origin."""
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


def _expand_compressed_tour(compressed_tour, path_lookup):
    """Expand every compressed edge into its real path inside the original graph."""
    expanded = [compressed_tour[0]]
    for u, v in _pairwise(compressed_tour):
        key = frozenset((u, v))
        real_path = path_lookup.get(key, [u, v])  # fallback to direct edge
        if real_path[0] == expanded[-1]:
            expanded.extend(real_path[1:])
        else:  # path stored in opposite direction
            expanded.extend(list(reversed(real_path))[1:])
    return expanded

# ---------------------------------------------------------------------
# 4. CNN routing                                      
# ---------------------------------------------------------------------

def cnn_routing(G, origin, blocked_edges):
    """Run the Christofides Nearest Neighbor algorithm.

    Parameters
    ----------
    G : networkx.Graph - complete metric graph (triangle inequality).
    origin : starting vertex s.
    blocked_edges : collection of unordered vertex pairs that are permanently
                    blocked (concealed from the algorithm until first probe).

    Returns
    -------
    route  - list of vertices (walk) starting and ending at *origin*.
    length - total distance travelled.
    """
    # Normalise blocked‑edge set
    blocked_edges = {_sorted_edge(u, v) for (u, v) in blocked_edges if u != v}

    # 1. Christofides tour
    P = get_christofides_tour(G, origin)

    # 2. Short‑Cut phase
    G_star, Us, P1 = _shortcut_phase(G, P, blocked_edges)

    # 3. Compression phase
    G_prime, path_lookup = _compress_phase(G_star, Us)

    # 4. Nearest‑Neighbor exploration on compressed graph
    P2_compressed = _nearest_neighbor_tour(G_prime, origin)
    P2 = _expand_compressed_tour(P2_compressed, path_lookup)

    # 5. Concatenate paths (skip duplicate origin in the middle and end)
    final_route = P1[:-1] + P2[1:]

    # 6. Total length (skip consecutive duplicates, use G as metric authority)
    total_len = 0.0
    prev = final_route[0]
    for cur in final_route[1:]:
        if cur == prev:
            continue  # ignore 0‑length self‑loops that may appear after expansion
        
        total_len += G[prev][cur]["weight"]
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
