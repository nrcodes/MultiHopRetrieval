from pathlib import Path
import spacy
from config import CONFIG

_nlp = spacy.load(CONFIG.spacy_model)

def load_documents(corpus_dir: str) -> list[dict]:
    return [
        {"doc_id": p.stem, "text": p.read_text().strip()}
        for p in sorted(Path(corpus_dir).glob("*.txt"))
    ]

def iter_sentences(documents: list[dict]):
    for doc in documents:
        for i, sent in enumerate(_nlp(doc["text"]).sents):
            yield {
                "doc_id": doc["doc_id"],
                "sent_id": i,
                "text": sent.text.strip(),
                "start": sent.start_char,
                "end": sent.end_char,
            }