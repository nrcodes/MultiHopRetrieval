import spacy
from src.models import Answer
from src.search import beam_search
from src.kg import canonical
from config import CONFIG

_nlp = spacy.load(CONFIG.spacy_model)

_AUX_SKIP = {"be", "have", "do", "will", "would", "can", "could",
             "may", "might", "must", "shall", "should", "did"}

def _query_lemmas(query: str) -> set[str]:
    return {
        t.lemma_.lower()
        for t in _nlp(query)
        if t.pos_ == "VERB"
        and t.lemma_.lower() not in _AUX_SKIP
    }

def _link_entities(G, query: str) -> list[str]:
    seeds = set()

    doc = _nlp(query)

    for chunk in doc.noun_chunks:
        key = canonical(chunk.text)
        if key in G:
            seeds.add(key)

    query_lower = query.lower()

    for node in G.nodes:
        if node in query_lower:
            seeds.add(node)

    return list(seeds)

def answer_query(G, query: str) -> Answer:
    seeds = _link_entities(G, query)
    if not seeds:
        return Answer(status="insufficient_evidence", query=query,
                      reason="no query entity linked to knowledge graph")

    paths = beam_search(G, seeds, _query_lemmas(query))
    if not paths or paths[0].score < CONFIG.confidence_threshold:
        return Answer(status="insufficient_evidence", query=query,
                      reason="no path above confidence threshold")

    top = paths[0]
    terminal = top.edges[-1]
    answer_display = G.nodes[terminal.to_node]["display"]

    # Ambiguity: second-best path is within ambiguity_band of top
    if len(paths) > 1 and paths[1].score >= top.score * (1 - CONFIG.ambiguity_band):
        return Answer(status="ambiguous", query=query, answer=answer_display,
                      confidence=round(top.score, 3), hops=len(top.edges),
                      chain=top.edges,
                      reason="multiple near-equal paths")

    return Answer(status="ok", query=query, answer=answer_display,
                  confidence=round(top.score, 3),
                  hops=len(top.edges), chain=top.edges)