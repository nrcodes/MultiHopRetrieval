import sys, json
from src.ingestion import load_documents, iter_sentences
from src.extraction import extract_triples
from src.kg import build_kg
from src.answer import answer_query
from config import CONFIG
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=CONFIG.corpus_dir)
    parser.add_argument("query", nargs="+")
    args = parser.parse_args()

    triples = []
    for sent in iter_sentences(load_documents(args.corpus)):
        triples.extend(extract_triples(sent))
    G = build_kg(triples)

    query = " ".join(args.query)
    print(json.dumps(answer_query(G, query).model_dump(), indent=2, default=str))

if __name__ == "__main__":
    main()