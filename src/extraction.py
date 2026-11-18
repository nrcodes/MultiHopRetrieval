import spacy
from src.models import Triple
from config import CONFIG

_nlp = spacy.load(CONFIG.spacy_model)
_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those"}
_TEMPORAL_PREPS = {"in", "on", "during", "since", "until", "for", "at"}

def _strip_det(text: str) -> str:
    toks = text.split()
    if toks and toks[0].lower() in _DETERMINERS:
        return " ".join(toks[1:])
    return text

def _np_of(token) -> str:
    for chunk in token.doc.noun_chunks:
        if token in chunk:
            return _strip_det(chunk.text)
    return _strip_det(token.text)

def _is_numeric(text: str) -> bool:
    return text.replace(",", "").replace(".", "").replace(" ", "").isdigit()

def _extract_object(verb):
    """Priority: direct object > passive agent ('by X') > non-temporal prep."""
    dobj = agent = prep = None
    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "oprd"):
            dobj = _np_of(child)
        elif child.dep_ == "agent":                    # passive "by X"
            for gc in child.children:
                if gc.dep_ == "pobj":
                    agent = _np_of(gc)
                    break
        elif child.dep_ == "prep" and prep is None:
            if child.text.lower() in _TEMPORAL_PREPS:
                continue
            for gc in child.children:
                if gc.dep_ == "pobj":
                    cand = _np_of(gc)
                    if _is_numeric(cand):
                        continue
                    prep = cand
                    break
    return dobj or agent or prep

def extract_triples(sentence: dict) -> list[Triple]:
    doc = _nlp(sentence["text"])
    triples: list[Triple] = []
    for verb in doc:
        if verb.pos_ != "VERB":
            continue
        subj = None
        for child in verb.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subj = _np_of(child)
                break
        if subj is None:
            continue
        obj = _extract_object(verb)
        if obj and subj.lower() != obj.lower():
            triples.append(Triple(
                subject=subj, predicate=verb.lemma_.lower(), object=obj,
                doc_id=sentence["doc_id"], sent_id=sentence["sent_id"],
                sentence=sentence["text"],
                char_span=(sentence["start"], sentence["end"]),
            ))
    return triples