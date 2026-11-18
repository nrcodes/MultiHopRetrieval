# Multi-Hop Contextual Document Retrieval Engine

A lightweight document retrieval system that identifies relationships distributed
across multiple documents and retrieves supporting evidence for complex multi-hop
queries. Built entirely with locally-executable open-source tools — no LLMs, no
paid APIs, no cloud infrastructure.

## What it does

Given a corpus of short documents and a natural-language question that requires
chaining facts across documents, the engine:

1. Extracts subject–verb–object triples from every sentence using spaCy's
   dependency parser.
2. Builds a provenance-annotated knowledge graph — every edge carries its
   `doc_id`, `sent_id`, `char_span`, and the original sentence.
3. Parses the query, links its entities to graph nodes, and extracts its verb
   lemmas.
4. Runs beam search over the graph to find the highest-scoring path up to 4 hops.
5. Returns the answer, a confidence score, and the full evidence chain — or
   explicitly abstains if no path clears the threshold.

## Architecture

```
Documents → spaCy (NER + dependency parse) → SVO triples with provenance
         → NetworkX MultiDiGraph
         → Query (entity linking + verb lemmas) → beam search
         → Path scoring by query-verb coverage → Evidence chain
         → CLI / FastAPI / Streamlit
```

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet')"
```

## Run

### CLI

```bash
python cli.py "Which product was launched by the company acquired by the organization that employed Employee X?"
```

### Streamlit UI

```bash
streamlit run streamlit_app.py
# opens http://localhost:8501 — shows the reasoning chain as a graph,
# the evidence table with source sentences, and confidence metrics
```

### FastAPI

```bash
uvicorn src.api:app --reload
# POST /query with {"query": "..."}
# interactive docs at http://localhost:8000/docs
```

### Use your own documents

Point the engine at any folder of `.txt` files:

```bash
python cli.py --corpus data/corpus "What did Acme Corp acquire?"
# → {"status":"ok","answer":"Beta Labs","hops":1,...}

python cli.py --corpus data/corpus "Which product was launched by the company acquired by the organization that employed Employee X?"
```

If the query's entities do not appear in the corpus, the engine returns
`insufficient_evidence` with a diagnostic reason.

## Example Output

Query: _"Which product was launched by the company acquired by the organization
that employed Employee X?"_

```json
{
  "status": "ok",
  "answer": "Product P",
  "confidence": 1.0,
  "hops": 3,
  "chain": [
    {
      "from_node": "employee x",
      "to_node": "company y",
      "predicate": "employ",
      "evidence": {
        "doc_id": "doc_a",
        "sent_id": 0,
        "sentence": "Employee X was employed by Company Y in 2020.",
        "char_span": [0, 45]
      }
    },
    {
      "from_node": "company y",
      "to_node": "company z",
      "predicate": "acquire",
      "evidence": {
        "doc_id": "doc_b",
        "sent_id": 0,
        "sentence": "Company Y acquired Company Z in 2022.",
        "char_span": [0, 37]
      }
    },
    {
      "from_node": "company z",
      "to_node": "product p",
      "predicate": "launch",
      "evidence": {
        "doc_id": "doc_c",
        "sent_id": 0,
        "sentence": "Company Z launched Product P in 2023.",
        "char_span": [0, 37]
      }
    }
  ]
}
```

Every hop in the chain carries `doc_id`, `sent_id`, `char_span`, and the
original sentence — the answer is fully traceable to source.

## Evaluation

Run:

```bash
python eval/run_eval.py
```

Results:

## query status answer evid

Which product was launched by the company acquired by the organizati ok Product P ✓
What product did Beta Labs launch? insufficient_evidence None ✗
What did Acme Corp acquire? ok Beta Labs ✓
Who is the CEO of Mars? insufficient_evidence — ✓
Which system was developed by the company acquired by Solaris Energy ok TurbineMax system ✓
What platform was launched by the company that partnered with Nova B insufficient_evidence None ✗
Which product did Gamma Industries launch? insufficient_evidence — ✓

Answer Exact Match: 5/7 (71%)
Evidence Recall: 3/5
Abstention Acc: 2/2

Test suite: `pytest -q` → 5 passed in 3.77s

## Design Decisions

**Dependency-based SVO over OpenIE libraries.** spaCy's dependency parse yields
(subject, verb, object) with character offsets attached to the same tokens, so
provenance is free. Stanford OpenIE requires a JVM and does not return span
offsets. For clean declarative sentences — the target corpus shape — dependency
extraction matches OpenIE quality.

**Passive-voice handling via the `agent` dependency.** "Employee X was employed
by Company Y" produces an `agent` dependency for `by Company Y`, not a `prep`
dependency. The extractor checks `agent` before `prep`, and skips temporal
prepositions (`in`, `on`, `during`) so `in 2020` never becomes a triple object.

**NetworkX over Neo4j.** At <100 documents, path enumeration is <5 ms in-memory.
Neo4j is slower at shallow depth and adds Docker orchestration for the reviewer.
The Cypher equivalent (`MATCH (a)-[*1..4]->(b)`) is listed in the Roadmap as the
production migration path. Critically, edge scoring — the actual algorithmic
work — stays in Python regardless of storage engine, because Cypher cannot
natively compute embedding or WordNet similarity per edge.

**Coverage-based path scoring over length penalty.** A common IR heuristic
(length penalty, `0.9^hop`) systematically prefers shorter paths — for a
3-verb query it would return the 1-hop intermediate node (`Company Y`) instead
of the correct 3-hop answer (`Product P`). Coverage scoring — the fraction of
distinct query verbs satisfied by the path — aligns path selection with query
intent. With 3 query verbs: 1-hop scores 0.33, 2-hop 0.67, 3-hop 1.0, so the
full chain wins.

**Abstention as a first-class state.** A retrieval engine that cannot say
"insufficient evidence" is a hallucination machine. Two abstention branches are
distinguished: `"no query entity linked"` (the query mentions nothing in the
graph) and `"no path above confidence threshold"` (the entity exists but no
connected path satisfies the query). Both are surfaced in the output reason
field.

## Limitations

1. **Coreference is not resolved.** "The paper" in `doc_e` does not link to
   "a research paper" in `doc_d` — they are separate graph nodes. Example 2
   from the assignment therefore fails at hop 2. Resolving this requires a
   coreference model (`fastcoref`, ~500 MB), deferred to v2.
2. **Nested prepositional phrases are not fully attributed.** "a research team
   at ABC University" extracts the team but loses the university affiliation.
3. **Predicate matching is lexical + WordNet.** "employed" vs "joined" scores
   below the 0.55 similarity floor unless the lemmas match or WordNet
   path-similarity is high enough. Handled in this corpus by matching query
   and document verbs.
4. **Single-sentence documents assumed.** Paragraph-level documents work
   (sentence segmentation is per-`.`), but co-reference across sentences within
   a paragraph is not resolved.

## Roadmap

- `fastcoref` for pronoun and definite-NP resolution (unlocks Example 2)
- `sentence-transformers` (`all-MiniLM-L6-v2`) for predicate scoring
- Neo4j + Cypher `MATCH (a)-[*1..4]->(b)` for path enumeration at scale
- BM25 passage pre-filter before beam search to reduce branching
- Negation detection ("Company Y did NOT acquire...")
