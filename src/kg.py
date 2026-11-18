import networkx as nx
from src.models import Triple

def canonical(name: str) -> str:
    """Entity normalization: lowercase, no leading determiner."""
    for det in ("the ", "a ", "an "):
        if name.lower().startswith(det):
            return name[len(det):].lower().strip()
    return name.lower().strip()

def build_kg(triples: list[Triple]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for t in triples:
        s, o = canonical(t.subject), canonical(t.object)
        if s not in G:
            G.add_node(s, display=t.subject)
        if o not in G:
            G.add_node(o, display=t.object)
        G.add_edge(s, o, predicate=t.predicate, evidence=t)
    return G