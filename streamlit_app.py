"""Interactive UI for the multi-hop retrieval engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd

from config import CONFIG
from src.ingestion import load_documents, iter_sentences
from src.extraction import extract_triples
from src.kg import build_kg
from src.answer import answer_query

st.set_page_config(page_title="Multi-Hop Retrieval", page_icon="🔗", layout="wide")


@st.cache_resource
def load_graph():
    triples = []
    for sent in iter_sentences(load_documents(CONFIG.corpus_dir)):
        triples.extend(extract_triples(sent))
    return build_kg(triples)


def build_dot(G, chain):
    lines = ["digraph G {", "rankdir=LR;",
             'node [shape=box, style=rounded, fontname="Helvetica"];',
             'edge [fontname="Helvetica", fontsize=10];']
    for edge in chain:
        if edge.direction == "fwd":
            a, b, label = edge.from_node, edge.to_node, edge.predicate
        else:                              # reverse: render in graph's true direction
            a, b, label = edge.to_node, edge.from_node, f"{edge.predicate} (rev)"
        a_disp = G.nodes[a]["display"]
        b_disp = G.nodes[b]["display"]
        style = "solid" if edge.direction == "fwd" else "dashed"
        lines.append(f'"{a_disp}" -> "{b_disp}" [label="{label}", style="{style}"];')
    lines.append("}")
    return "\n".join(lines)


def chain_dataframe(G, chain):
    rows = []
    for hop, edge in enumerate(chain, 1):
        src = G.nodes[edge.from_node]["display"]
        tgt = G.nodes[edge.to_node]["display"]
        arrow = "→" if edge.direction == "fwd" else "←"
        rows.append({
            "hop": hop,
            "from": src,
            "predicate": edge.predicate,
            "to": tgt,
            "direction": arrow,
            "source": f"{edge.evidence.doc_id}#{edge.evidence.sent_id}",
            "evidence": edge.evidence.sentence,
        })
    return pd.DataFrame(rows)


def main():
    st.title("🔗 Multi-Hop Contextual Document Retrieval")
    st.caption("Provenance-annotated knowledge graph · no LLMs · runs locally")

    G = load_graph()

    examples = [
        "Which product was launched by the company acquired by the organization that employed Employee X?",
        "What did Acme Corp acquire?",
        "What product did Beta Labs launch?",
        "Who is the CEO of Mars?",
    ]

    with st.sidebar:
        st.header("Corpus")
        st.metric("Documents", len(load_documents(CONFIG.corpus_dir)))
        st.metric("Graph nodes", G.number_of_nodes())
        st.metric("Graph edges", G.number_of_edges())
        st.divider()
        st.subheader("Try an example")
        for i, ex in enumerate(examples):
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                st.session_state.query = ex
                st.session_state.auto_run = True
        st.divider()
        st.caption(
            "Tip: unanswerable queries (e.g. Mars) return "
            "`insufficient_evidence` — that's a feature, not a bug."
        )

    if "query" not in st.session_state:
        st.session_state.query = examples[0]

    query = st.text_input("Query", key="query")
    run = st.button("Retrieve", type="primary")
    if st.session_state.get("auto_run"):
        st.session_state.auto_run = False
        run = True

    if not run:
        return

    with st.spinner("Searching knowledge graph…"):
        result = answer_query(G, query)

    if result.status == "ok":
        c1, c2, c3 = st.columns(3)
        c1.metric("Answer", result.answer)
        c2.metric("Confidence", f"{result.confidence:.2f}")
        c3.metric("Hops", result.hops)

        st.subheader("Reasoning chain")
        st.graphviz_chart(build_dot(G, result.chain), use_container_width=True)

        st.subheader("Evidence (traceable to source)")
        st.dataframe(chain_dataframe(G, result.chain),
                     use_container_width=True, hide_index=True)

        with st.expander("Raw JSON response"):
            st.json(result.model_dump())

    elif result.status == "ambiguous":
        st.warning(f"Ambiguous — best guess: **{result.answer}** ({result.reason})")
        with st.expander("Raw JSON response"):
            st.json(result.model_dump())

    else:
        st.error(f"Insufficient evidence — {result.reason}")
        with st.expander("Raw JSON response"):
            st.json(result.model_dump())


if __name__ == "__main__":
    main()