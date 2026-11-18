from src.answer import answer_query

def test_example_1_full_chain(graph):
    r = answer_query(graph, "Which product was launched by the company acquired "
                            "by the organization that employed Employee X?")
    assert r.status == "ok"
    assert "product p" in r.answer.lower()
    assert r.hops == 3
    assert all(e.evidence.doc_id.startswith("doc_") for e in r.chain)

def test_abstains_on_unknown(graph):
    r = answer_query(graph, "Who is the CEO of Mars?")
    assert r.status == "insufficient_evidence"