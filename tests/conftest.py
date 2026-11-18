import pytest
from src.ingestion import load_documents, iter_sentences
from src.extraction import extract_triples
from src.kg import build_kg

@pytest.fixture(scope="session")
def graph():
    triples = []
    for sent in iter_sentences(load_documents("data/corpus")):
        triples.extend(extract_triples(sent))
    return build_kg(triples)