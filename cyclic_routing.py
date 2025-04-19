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
    """Return a 1.5‑approximate TSP tour using Christofides (via networkx if
    available, else a naïve greedy fallback)."""

    if nx is not None and hasattr(nx.algorithms.approximation, "christofides"):
        from networkx.algorithms.approximation.traveling_salesman import (
            traveling_salesman_problem as _tsp,
        )

        tour: List[int] = _tsp(
            G,
            cycle=True,
            weight="weight",
            method=nx.algorithms.approximation.christofides,
            start=origin,
        )
        return tour[:-1]  # drop duplicate origin added by networkx

    # ---------------- fallback (greedy nearest‑neighbour) ----------------
    unvisited = set(G.nodes())
    unvisited.remove(origin)
    tour = [origin]
    current = origin
    while unvisited:
        next_v = min(unvisited, key=lambda v: G[current][v]["weight"])
        tour.append(next_v)
        unvisited.remove(next_v)
        current = next_v
    return tour  # origin not repeated – cyclic handling done by caller

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

        # --- direct edge blocked -> try a two‑hop detour via already‑visited
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
