import networkx as nx
from src.models import Edge, Path
from src.scoring import predicate_score
from config import CONFIG

def _neighbors(G, node):
    for _, tgt, data in G.out_edges(node, data=True):
        yield tgt, data, "fwd"
    for src, _, data in G.in_edges(node, data=True):
        yield src, data, "rev"

def _path_score(edges, query_lemmas):
    if not query_lemmas:
        return 0.0
    used = set()
    for e in edges:
        for ql in query_lemmas:
            if predicate_score({ql}, e.predicate) > 0:
                used.add(ql)
                break
    return len(used) / len(query_lemmas)

def beam_search(G: nx.MultiDiGraph, seeds: list[str],
                query_lemmas: set[str]) -> list[Path]:
    frontier = [(seed, []) for seed in seeds]
    completed: list[Path] = []
    for _hop in range(CONFIG.max_hops):
        candidates = []
        for node, edges in frontier:
            visited = {node} | {e.from_node for e in edges} | {e.to_node for e in edges}
            for nbr, data, direction in _neighbors(G, node):
                if nbr in visited:
                    continue
                ps = predicate_score(query_lemmas, data["predicate"])
                if ps == 0:
                    continue
                edge = Edge(from_node=node, to_node=nbr,
                            predicate=data["predicate"],
                            evidence=data["evidence"], direction=direction)
                new_edges = edges + [edge]
                candidates.append((nbr, new_edges, _path_score(new_edges, query_lemmas)))
        if not candidates:
            break
        candidates.sort(key=lambda c: (c[2], len(c[1])), reverse=True)
        top = candidates[:CONFIG.beam_width]
        for _, e, s in top:
            completed.append(Path(edges=e, score=s))
        frontier = [(n, e) for n, e, _ in top]
    return sorted(completed, key=lambda p: (p.score, len(p.edges)), reverse=True)