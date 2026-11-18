from fastapi import FastAPI
from pydantic import BaseModel
from src.ingestion import load_documents, iter_sentences
from src.extraction import extract_triples
from src.kg import build_kg
from src.answer import answer_query
from src.models import Answer
from config import CONFIG

app = FastAPI(title="Multi-Hop Retrieval", version="1.0.0")

_G = None

def get_graph():
    global _G
    if _G is None:
        triples = []
        for sent in iter_sentences(load_documents(CONFIG.corpus_dir)):
            triples.extend(extract_triples(sent))
        _G = build_kg(triples)
    return _G

class QueryRequest(BaseModel):
    query: str

@app.post("/query", response_model=Answer)
async def query(q: QueryRequest) -> Answer:
    return answer_query(get_graph(), q.query)

@app.get("/health")
async def health():
    G = get_graph()
    return {"status": "ok", "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}