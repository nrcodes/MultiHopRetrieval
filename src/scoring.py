from functools import lru_cache
from nltk.corpus import wordnet as wn
from config import CONFIG

@lru_cache(maxsize=4096)
def _wordnet_sim(a: str, b: str) -> float:
    sa, sb = wn.synsets(a), wn.synsets(b)
    if not sa or not sb:
        return 0.0
    return max((s1.path_similarity(s2) or 0.0)
               for s1 in sa[:3] for s2 in sb[:3])

def predicate_score(query_lemmas: set[str], predicate: str) -> float:
    if predicate in query_lemmas:
        return 1.0
    best = max((_wordnet_sim(q, predicate) for q in query_lemmas), default=0.0)
    return best if best >= CONFIG.predicate_similarity_floor else 0.0