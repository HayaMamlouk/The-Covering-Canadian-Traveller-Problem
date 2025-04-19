import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem

def get_christofides_tour(G, start_node):
    tour = traveling_salesman_problem(G, cycle=True, weight='cost')
    if start_node in tour:
        idx = tour.index(start_node)
        tour = tour[idx:] + tour[:idx]
    return tour[:-1]  # exclude return to start


def cr_algorithm(G, start_node, tour, blocked):
    """
    Cyclic‑Routing (CR) algorithm — final, faithful version.

    Parameters
    ----------
    G        : networkx.Graph        (complete graph; weights unused)
    start_node : str                 starting vertex (v1)
    tour     : list[str]             fixed circular order [v1, v2, …, vn]
    blocked  : set[tuple[str,str]]   blocked edges stored as *sorted* tuples

    Returns
    -------
    walk     : list[tuple[str,str]]  directed traversal edges
    """
    # -----------------------------------------------------------------
    sEdge = lambda a, b: tuple(sorted((a, b)))       # canonical edge

    def cw_between(a, b):
        """Vertices strictly CW between a and b along the tour."""
        res, n = [], len(tour)
        ia, ib = tour.index(a), tour.index(b)
        k = (ia + 1) % n
        while k != ib:
            res.append(tour[k])
            k = (k + 1) % n
        return res

    def agenda_list(cur, direction, unvisited):
        """
        Starting with the vertex NEXT to 'cur' in the given direction,
        list unvisited vertices encountered while looping once around.
        """
        n      = len(tour)
        idx    = tour.index(cur)
        order  = tour if direction == 1 else list(reversed(tour))
        seq    = []
        k      = (order.index(cur) + 1) % n
        start  = order[k]
        while True:
            if order[k] in unvisited:
                seq.append(order[k])
            k = (k + 1) % n
            if order[k] == start:
                break
        return seq
    # -----------------------------------------------------------------

    visited, unvisited = {start_node}, set(G.nodes) - {start_node}
    cur, direction     = start_node, 1          # 1 = CW, −1 = CCW
    walk               = []
    round_no           = 1

    while unvisited:
        todo = agenda_list(cur, direction, unvisited)
        print(f"\n[Round {round_no}] dir={'CW' if direction==1 else 'CCW'}")
        if not todo:
            break

        expected_last = todo[-1]
        progress      = False
        i             = 0

        while i < len(todo):
            tgt   = todo[i]
            e_dir = sEdge(cur, tgt)

            # Case 1: direct edge available
            if e_dir not in blocked:
                walk.append((cur, tgt))
                visited.add(tgt); unvisited.remove(tgt)
                cur = tgt
                progress = True
                i += 1
                continue

            # Case 2: direct edge blocked → detour via visited nodes on arc
            print(f"  Blocked {cur} → {tgt}")
            blocked.add(e_dir)

            arc   = cw_between(cur, tgt) if direction == 1 else cw_between(tgt, cur)[::-1]
            detour_done = False
            for w in arc:
                if w not in visited:
                    continue
                e1, e2 = sEdge(cur, w), sEdge(w, tgt)
                if e1 not in blocked and e2 not in blocked:
                    print(f"    Detour {cur} → {w} → {tgt}")
                    walk.extend([(cur, w), (w, tgt)])
                    visited.add(tgt); unvisited.remove(tgt)
                    cur = tgt
                    progress = True
                    detour_done = True
                    break

            if not detour_done:
                print(f"    Skip {tgt}")
            i += 1

        # Flip direction if we should have ended at expected_last but didn’t
        if progress and cur != expected_last:
            direction *= -1
            print(f"  Flip direction (ended at {cur}, expected {expected_last})")

        if not progress:
            print("[!] No progress – remaining nodes unreachable.")
            break

        round_no += 1
        if round_no > 100:      # safety cap
            print("[ABORT] Too many rounds.")
            break

    # -----------------------------------------------------------------
    # Return to start (respects tour order for the intermediate)
    # -----------------------------------------------------------------
    direct = sEdge(cur, start_node)
    if direct not in blocked:
        walk.append((cur, start_node))
    else:
        found = False
        # 1) look clockwise arc
        for w in cw_between(cur, start_node):
            if (w in visited and
                sEdge(cur, w) not in blocked and
                sEdge(w, start_node) not in blocked):
                walk.extend([(cur, w), (w, start_node)])
                found = True
                break
        # 2) look counter‑clockwise arc
        if not found:
            for w in cw_between(start_node, cur):
                if (w in visited and
                    sEdge(cur, w) not in blocked and
                    sEdge(w, start_node) not in blocked):
                    walk.extend([(cur, w), (w, start_node)])
                    found = True
                    break
        # 3) final fallback: any visited vertex
        if not found:
            for w in visited:
                if (sEdge(cur, w) not in blocked and
                    sEdge(w, start_node) not in blocked):
                    walk.extend([(cur, w), (w, start_node)])
                    found = True
                    break
        if not found:
            print("[!] Cannot return to start – all paths blocked.")

    return walk







