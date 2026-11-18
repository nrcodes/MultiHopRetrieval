import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.ingestion import load_documents, iter_sentences
from src.extraction import extract_triples
from src.kg import build_kg
from src.answer import answer_query

def main():
    triples = []
    for sent in iter_sentences(load_documents("data/corpus")):
        triples.extend(extract_triples(sent))
    G = build_kg(triples)

    qs = json.load(open("eval/questions.json"))
    answerable = [q for q in qs if q["gold"] is not None]
    abstain = [q for q in qs if q["gold"] is None]

    em = hits = evidence_hits = evidence_total = 0
    print(f"{'query':<70} {'status':<22} {'answer':<20} {'evid':<6}")
    print("-" * 125)
    for q in qs:
        r = answer_query(G, q["query"])
        if q["gold"] is None:
            ok = r.status == "insufficient_evidence"
            em += ok
            print(f"{q['query'][:68]:<70} {r.status:<22} {'—':<20} {'✓' if ok else '✗'}")
            continue
        correct = r.answer is not None and q["gold"].lower() in r.answer.lower()
        em += correct
        doc_ids = {e.evidence.doc_id for e in r.chain}
        expected = set(q["gold_docs"])
        ev_ok = expected.issubset(doc_ids)
        evidence_hits += ev_ok
        evidence_total += 1
        print(f"{q['query'][:68]:<70} {r.status:<22} {str(r.answer):<20} {'✓' if ev_ok else '✗'}")

    print(f"\nAnswer Exact Match: {em}/{len(qs)}  ({em/len(qs):.0%})")
    print(f"Evidence Recall:    {evidence_hits}/{evidence_total}")
    print(f"Abstention Acc:     {sum(1 for q in abstain if answer_query(G, q['query']).status == 'insufficient_evidence')}/{len(abstain)}")

if __name__ == "__main__":
    main()