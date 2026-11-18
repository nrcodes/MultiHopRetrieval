from dataclasses import dataclass, field

@dataclass(frozen=True)
class Config:
    corpus_dir: str = "data/corpus"
    spacy_model: str = "en_core_web_sm"
    max_hops: int = 4
    beam_width: int = 8
    confidence_threshold: float = 0.6
    ambiguity_band: float = 0.05       
    predicate_similarity_floor: float = 0.55
    length_penalty: float = 0.9

CONFIG = Config()