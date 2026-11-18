# debug.py
from config import CONFIG
from src.ingestion import load_documents, iter_sentences
from src.extraction import extract_triples
from src.kg import build_kg, canonical
from src.answer import _link_entities, _query_lemmas
import spacy

print("=== 1. Corpus ===")
docs = load_documents(CONFIG.corpus_dir)
print(f"Loaded {len(docs)} docs")
for d in docs:
    print(f"  {d['doc_id']}: {d['text'][:80]}")

print("\n=== 2. Extracted Triples ===")
triples = []
for sent in iter_sentences(docs):
    triples.extend(extract_triples(sent))
print(f"Total: {len(triples)}")
for t in triples:
    print(f"  ({t.subject}) --[{t.predicate}]--> ({t.object})   [{t.doc_id}]")

print("\n=== 3. Graph Nodes ===")
G = build_kg(triples)
print(f"Nodes ({G.number_of_nodes()}): {list(G.nodes)}")
print(f"Edges ({G.number_of_edges()})")

print("\n=== 4. Query Analysis ===")
query = "Which product was launched by the company acquired by the organization that employed Employee X?"
nlp = spacy.load(CONFIG.spacy_model)
print(f"Noun chunks:        {[c.text for c in nlp(query).noun_chunks]}")
print(f"Canonical chunks:   {[canonical(c.text) for c in nlp(query).noun_chunks]}")
print(f"Seeds linked to KG: {_link_entities(G, query)}")
print(f"Query lemmas:       {_query_lemmas(query)}")

print("\n=== 5. Direct Edge Check ===")
key = "employee x"
if key in G:
    print(f"Out-edges from '{key}':")
    for _, tgt, data in G.out_edges(key, data=True):
        print(f"  → {tgt}  via '{data['predicate']}'")
else:
    print(f"'{key}' NOT in graph nodes")