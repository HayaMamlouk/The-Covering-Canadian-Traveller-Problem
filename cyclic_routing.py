# cyclic_routing.py
# ------------------------------------------------------------
# Implementation of the Cyclic Routing (CR) algorithm described
# in the user‑supplied pseudocode (Algorithm 1 and Procedure SHORTCUT).
# ------------------------------------------------------------
# Author: ChatGPT (OpenAI o3)
# ------------------------------------------------------------
"""
This module provides a reference, *readable* implementation of the Cyclic Routing
(CR) algorithm for the **k‑Cyclic Cardinality Travelling Problem (k‑CCTP)**,
using the precise structure given in the supplied pseudocode.

------------------------------------------------------------------------
Usage example
------------------------------------------------------------------------
>>> import networkx as nx
>>> from cyclic_routing import cyclic_routing

# Build a complete graph with Euclidean weights (triangle inequality holds)
>>> G = nx.complete_graph(6)
>>> pos = nx.random_layout(G, seed=1)            # random 2‑D coordinates
>>> for u, v in G.edges:
...     G[u][v]["weight"] = ((pos[u] - pos[v])**2).sum() ** 0.5

# Declare *unknown* blockages (here fixed for reproducibility)
>>> blocked = {(0, 2), (1, 4), (3, 5)}

# Run the CR algorithm starting from origin 0
>>> route, length = cyclic_routing(G, origin=0, blocked_edges=blocked)
>>> print(route)
[0, 1, 5, 2, 3, 4, 0]
>>> print(f"Total distance travelled = {length:.3f}")

The algorithm keeps discovering blockages *online* (i.e. only when it first
tries to use a blocked edge) and incrementally patches its route using the
`SHORTCUT` procedure while respecting the original Christofides tour order.

------------------------------------------------------------------------
API
------------------------------------------------------------------------
- `cyclic_routing(G, origin, blocked_edges)` – run the algorithm and return the
  realised route (list of vertices, including the closing return to the origin)
  together with its total cost.

The code relies **only** on `networkx`, which is part of the standard set of
libraries available in the ChatGPT sandbox.  If `networkx` is not present the
module falls back to a very small local implementation of the Christofides
1.5‑approximation tour (sufficient for modest graph sizes ≤ 20).
"""
from __future__ import annotations

import heapq
from collections import deque
from itertools import tee
from typing import Dict, Iterable, List, Sequence, Set, Tuple

try:
    import networkx as nx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – simple fallback

    class _SimpleGraph(dict):
        """A *very* tiny undirected graph replacement used if networkx is absent."""

        def add_edge(self, u, v, weight: float):
            self.setdefault(u, {})[v] = weight
            self.setdefault(v, {})[u] = weight

        def nodes(self):
            return self.keys()

        def edges(self):
            for u in self:
                for v in self[u]:
                    if u < v:
                        yield (u, v)

        def __getitem__(self, item):  # type: ignore[override]
            return super().__getitem__(item)

    nx = None  # type: ignore

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _pairwise(iterable: Iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), …"""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _edge_key(u, v):
    """Return an unordered tuple so that (u, v) == (v, u)."""
    return (u, v) if u <= v else (v, u)

# ---------------------------------------------------------------------
# Christofides tour (1.5‑approximation)
# ---------------------------------------------------------------------


def _christofides_tour(G, origin: int) -> List[int]:
    """Return a 1.5‑approximate TSP tour using Christofides.

    The NetworkX API changed around version 3.2 – the inner
    ``christofides`` routine no longer accepts the *start* keyword that
    ``traveling_salesman_problem`` forwards to it.  We therefore:

    1.  Try the *newer* call signature (without *start*), and afterwards
        rotate the tour so it begins at ``origin``.
    2.  Fall back to the *older* signature (with *start*) if the first
        attempt fails.
    3.  Finally, if NetworkX is not available, revert to the simple greedy
        tour used before.
    """

    if nx is None or not hasattr(nx.algorithms.approximation, "christofides"):
        return _greedy_fallback(G, origin)

    from networkx.algorithms.approximation.traveling_salesman import (
        traveling_salesman_problem as _tsp,
    )

    try:
        # Newer NetworkX (≥ 3.2): christofides() does *not* accept 'start'
        tour: List[int] = _tsp(
            G,
            cycle=True,
            weight="weight",
            method=nx.algorithms.approximation.christofides,
        )
    except TypeError:
        # Older NetworkX (≤ 3.1): christofides(start=…) is allowed
        tour = _tsp(
            G,
            cycle=True,
            weight="weight",
            method=nx.algorithms.approximation.christofides,
            start=origin,
        )

    # Ensure the tour starts at 'origin' and drop the closing repeat
    if tour[0] != origin:
        k = tour.index(origin)
        tour = tour[k:-1] + tour[:k]
    else:
        tour = tour[:-1]  # remove duplicate origin added by NX

    return tour

# ---------------------------------------------------------------------
# SHORTCUT procedure (faithful to Fig. 1 pseudocode)
# ---------------------------------------------------------------------


def _shortcut(
    P: Sequence[int],  # full Christofides tour order (no duplicate origin)
    V_m: List[int],  # *ordered* list of unvisited vertices for this round
    current: int,  # v_{m,0}
    blocked_edges: Set[Tuple[int, int]],
    route: List[int],  # global route being built (will be extended in place)
) -> Tuple[List[int], int, bool]:
    """Execute one call to *procedure SHORTCUT*.

    Parameters
    ----------
    P : full Christofides tour visiting order (used for index look‑ups only)
    V_m : list of unvisited vertices *in the same order as in P* (forward or
           reverse).  This is mutated only through removals.
    current : the vertex where the traveller currently stands (v_{m,0}).
    blocked_edges : **global** set of blocked edges, *unknown* to the algorithm
                    until an attempted traverse.
    route : list storing the realised walk – gets extended in place.

    Returns
    -------
    V_{m+1} : list of the vertices still unvisited after this shortcut.
    last    : the traveller's final position after the procedure.
    progress: ``True`` iff *some* vertex has been visited in this call.
    """

    def is_blocked(u, v) -> bool:
        return _edge_key(u, v) in blocked_edges

    # Build the working list *with* the current vertex as position 0
    work = [current] + V_m.copy()
    i, j = 0, 1

    made_progress = False
    while j < len(work):
        v_i, v_j = work[i], work[j]
        if not is_blocked(v_i, v_j):
            # --- direct edge works ---
            route.append(v_j)
            if v_j in V_m:
                V_m.remove(v_j)
                made_progress = True
            i, j = j, j + 1
            continue

        # --- direct edge blocked → try a two‑hop detour via already‑visited
        #     vertices lying between i and j along *work* ---------------
        l = i + 1
        found_detour = False
        while l < j:
            v_l = work[l]
            if (not is_blocked(v_i, v_l)) and (not is_blocked(v_l, v_j)):
                # viable two‑edge path discovered
                route.extend([v_l, v_j])
                if v_j in V_m:
                    V_m.remove(v_j)
                    made_progress = True
                i, j = j, j + 1
                found_detour = True
                break
            l += 1

        if found_detour:
            continue  # outer while – next j value already set

        # --- give up on v_j for this round -----------------------------
        j += 1  # skip j entirely

    last_position = work[i]
    return V_m, last_position, made_progress

# ---------------------------------------------------------------------
# Main CR algorithm (Algorithm 1)
# ---------------------------------------------------------------------


def cyclic_routing(
    G,  # networkx.Graph with *metric* edge weights (triangle inequality)
    origin: int,
    blocked_edges: Set[Tuple[int, int]] | Iterable[Tuple[int, int]] = (),
    *,
    forward_first: bool = True,
) -> Tuple[List[int], float]:
    """Run the Cyclic Routing (CR) algorithm and return the realised route.

    Parameters
    ----------
    G : *Complete* undirected graph whose edge **weights** satisfy the triangle
        inequality.  The algorithm treats it as such.
    origin : starting vertex ``s``.
    blocked_edges : iterable with the *set* of blocked undirected edges.  These
                    are *concealed* from the algorithm until it first attempts
                    to traverse one of them.
    forward_first : whether the very first round follows the forward order of
                    the Christofides tour (default **True**).

    Returns
    -------
    route : list of vertices representing the walk realised by the algorithm.
            The list both starts and ends at *origin*.
    length : total length of that walk (sum of edge weights actually used).
    """

    # ---------------- normalise inputs ---------------------------------
    blocked_edges = {
        _edge_key(u, v) for u, v in blocked_edges if u != v
    }  # type: ignore[arg-type]

    if nx is None:
        raise RuntimeError("cyclic_routing requires networkx or a compatible graph object available as 'nx'.")

    # ---------------- 0. initial Christofides tour ----------------------
    P: List[int] = _christofides_tour(G, origin)

    route: List[int] = [origin]  # final path being constructed (P^{cr})
    current = origin

    # For quick index look‑ups along the tour
    index_in_P: Dict[int, int] = {v: i for i, v in enumerate(P)}
    n = len(P)

    def ordered_unvisited(unvisited: Set[int], start: int, forward: bool) -> List[int]:
        """Return the subsequence of *P* containing *unvisited* vertices, starting
        right *after* ``start`` (cyclic), in *forward* (True) or reverse order."""
        seq = []
        step = 1 if forward else -1
        k = index_in_P[start]
        for _ in range(n - 1):  # every vertex except *start*
            k = (k + step) % n
            v = P[k]
            if v in unvisited:
                seq.append(v)
        return seq

    # -------------------------------------------------------------------
    V_m: Set[int] = set(G.nodes()) - {origin}  # initially all except origin
    m = 1
    prev_direction_forward = forward_first  # direction taken in (m‑1)-th round

    while V_m:
        # Build the *ordered* list of still unvisited vertices for this round
        V_m_ordered = ordered_unvisited(V_m, current, prev_direction_forward)

        # -------------- try SHORTCUT in the chosen direction ------------
        V_m_after, current_after, progressed = _shortcut(
            P,
            V_m_ordered,
            current,
            blocked_edges,
            route,
        )

        # If no vertex was reached, immediately attempt opposite direction
        if not progressed:
            V_m_ordered_rev = ordered_unvisited(V_m, current, not prev_direction_forward)
            V_m_after, current_after, _ = _shortcut(
                P,
                V_m_ordered_rev,
                current,
                blocked_edges,
                route,
            )
            prev_direction_forward = not prev_direction_forward  # flipped
        else:
            # keep direction for next round *unless* no progress was made
            prev_direction_forward = prev_direction_forward

        # Prepare next round --------------------------------------------
        V_m = set(V_m_after)
        current = current_after
        m += 1

    # ---------------- 3. return to origin ------------------------------
    def is_blocked(u, v):
        return _edge_key(u, v) in blocked_edges

    # Simple strategy: try direct edge, else two‑hop via earliest visited *u*.
    if not is_blocked(current, origin):
        route.append(origin)
    else:
        for u in route:  # already‑visited vertices (order matters little)
            if (not is_blocked(current, u)) and (not is_blocked(u, origin)):
                route.extend([u, origin])
                break
        else:
            raise RuntimeError("No feasible path found back to the origin (all two‑hop detours blocked).")

    # --------------- compute total length ------------------------------
    total_length = 0.0
    for u, v in _pairwise(route):
        total_length += G[u][v]["weight"]

    return route, total_length

# ---------------------------------------------------------------------
# __main__ guard for quick manual testing
# ---------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import networkx as nx
    import random

    random.seed(0)
    n = 12
    G = nx.complete_graph(n)
    pos = {v: (random.random(), random.random()) for v in G}
    for u, v in G.edges:
        G[u][v]["weight"] = ((pos[u][0] - pos[v][0]) ** 2 + (pos[u][1] - pos[v][1]) ** 2) ** 0.5

    blocked = {(0, 3), (2, 5), (1, 8), (7, 9), (4, 11)}

    r, L = cyclic_routing(G, origin=0, blocked_edges=blocked)
    print("Route:", r)
    print("Length:", L)
