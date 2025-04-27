# MAMLOUK Haya [21107689]
# OZGENC Doruk [21113927]

import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem

# ---------------------------------------------------------------------
# Utilities                                                           
# ---------------------------------------------------------------------
def _sorted_edge(u, v) :
    """Return a sorted tuple of the edge (u, v) to ensure undirectedness."""
    return tuple(sorted((u, v)))

def cw_between(a , b, tour):
    """Return nodes encountered moving CW from `a` to `b` (exclusive)."""
    res = []
    n = len(tour)
    i = (tour.index(a) + 1) % n
    # collect until we reach b
    while tour[i] != b:
        res.append(tour[i])
        i = (i + 1) % n
    return res

def agenda_list(cur, direction, unvisited, tour):
    """
    Create the list of unvisited nodes encountered by making one full lap
    around the tour, starting immediately after cur, in the given
    direction (1=CW, -1=CCW).

    Returns:
        List of nodes (subset of unvisited) in the order they appear.
    """
    # choose traversal order
    order = tour if direction == 1 else list(reversed(tour))
    seq = []
    n = len(order)
    # start just after current node
    start_idx = order.index(cur)
    idx = (start_idx + 1) % n
    first = order[idx]
    # walk exactly one lap
    while True:
        node = order[idx]
        # include if not visited yet
        if node in unvisited:
            seq.append(node)
        idx = (idx + 1) % n
        if order[idx] == first:
            break
    return seq

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

# ----------------------------------------------------------------------
# Core CR Algorithm
# ----------------------------------------------------------------------
def cr_algorithm(G, start_node, tour, blocked) :
    """
    Perform the Cyclic-Routing (CR) traversal on G using a fixed circular
    tour and avoiding edges in blocked. Returns the list of directed
    edges (u, v) that form the walk, finishing back at start_node.
    """
    # --- Initialization ---
    visited = {start_node}             # nodes we've reached
    unvisited = set(G.nodes) - visited # nodes still to reach
    cur = start_node                   # current node
    direction = 1                      # traversal direction: 1=CW, -1=CCW
    walk = []   # edges of the CR walk
    round_no = 1                       # iteration counter

    # --- Main CR loop: visit all nodes ---
    while unvisited :
        # plan next candidates
        todo = agenda_list(cur, direction, unvisited, tour)
        if not todo:
            # no unvisited nodes reachable 
            break

        expected_last = todo[-1]  # node we'd finish on if everything goes CW
        progressed = False         # did we visit anyone this round?

        # try each target in agenda
        for tgt in todo:
            edge = _sorted_edge(cur, tgt)

            # --- Case 1: direct edge is free ---
            if edge not in blocked:
                walk.append((cur, tgt))
                visited.add(tgt)
                unvisited.remove(tgt)
                cur = tgt
                progressed = True
                continue

            # --- Case 2: direct edge blocked ---
            # mark blocked and attempt a two-hop detour via visited nodes
            blocked.add(edge)
            arc = (cw_between(cur, tgt, tour) if direction == 1
                   else list(reversed(cw_between(tgt, cur, tour))))
            for mid in arc:
                if mid in visited:
                    e1, e2 = _sorted_edge(cur, mid), _sorted_edge(mid, tgt)
                    if e1 not in blocked and e2 not in blocked:
                        # take detour cur->mid->tgt
                        walk.extend([(cur, mid), (mid, tgt)])
                        visited.add(tgt)
                        unvisited.remove(tgt)
                        cur = tgt
                        progressed = True
                        break
            # if no detour found, skip tgt

            # stop early if all visited
            if not unvisited:
                break

        # if we progressed but didn't end on expected_last, flip direction
        if progressed and cur != expected_last:
            direction *= -1
        # if no progress at all, abort
        if not progressed:
            break

        round_no += 1

    # --- Return to start_node ---
    final_edge = _sorted_edge(cur, start_node)
    if final_edge not in blocked:
        # direct return
        walk.append((cur, start_node))
    else:
        # try one- or two-hop via any visited node
        done = False
        for mid in cw_between(cur, start_node, tour) + cw_between(start_node, cur, tour):
            if mid in visited:
                e1, e2 = _sorted_edge(cur, mid), _sorted_edge(mid, start_node)
                if e1 not in blocked and e2 not in blocked:
                    walk.extend([(cur, mid), (mid, start_node)])
                    done = True
                    break
        # final fallback: direct via any visited
        if not done:
            for mid in visited:
                e1, e2 = _sorted_edge(cur, mid), _sorted_edge(mid, start_node)
                if e1 not in blocked and e2 not in blocked:
                    walk.extend([(cur, mid), (mid, start_node)])
                    break

    return walk

# ----------------------------------------------------------------------
# Entry Point: cyclic_routing
# ----------------------------------------------------------------------
def cyclic_routing(G, origin, blocked_edges) :
    """
    Compute a cyclic route using Christofides tour + CR algorithm.
    """
    blocked = {tuple(sorted(e)) for e in blocked_edges}
    tour = get_christofides_tour(G, origin)
    # tour = list(G.nodes)  # To test pdf example
    walk = cr_algorithm(G, origin, tour, blocked)
    route = [origin] + [v for _, v in walk]
    length = sum(G[u][v]['weight'] for u, v in walk)
    return route, length


# ---------------------------------------------------------------------
# __main__ guard for quick manual testing
# ---------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import random

    random.seed(0)
    n = 16
    G = nx.complete_graph(n)
    pos = {v: (random.random(), random.random()) for v in G}
    for u, v in G.edges:
        G[u][v]["weight"] = ((pos[u][0] - pos[v][0]) ** 2 + (pos[u][1] - pos[v][1]) ** 2) ** 0.5

    blocked = {(2,3),(2,4),(6,7),(8,9),(11,12),(11,13),(3,15),
               (3,4),(7,9),(12,13),(4,13),(4,9),(9,12),(0,13)}

    r, L = cyclic_routing(G, origin=0, blocked_edges=blocked)
    print("Route:", r)
    print("Length:", L)
