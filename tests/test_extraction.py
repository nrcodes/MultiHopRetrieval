from src.extraction import extract_triples

def test_simple_svo():
    sent = {"doc_id": "d", "sent_id": 0,
            "text": "Company Y acquired Company Z in 2022.", "start": 0, "end": 40}
    triples = extract_triples(sent)
    assert any(t.subject.lower() == "company y" and t.predicate == "acquire"
               and t.object.lower() == "company z" for t in triples)

def test_skips_self_reference():
    sent = {"doc_id": "d", "sent_id": 0, "text": "It launched it.", "start": 0, "end": 14}
    assert extract_triples(sent) == []