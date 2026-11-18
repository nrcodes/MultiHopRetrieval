from src.search import beam_search
from src.kg import canonical

def test_multi_hop_example_1(graph):
    seeds = [canonical("Employee X")]
    lemmas = {"employ", "acquire", "launch"}
    paths = beam_search(graph, seeds, lemmas)
    assert paths
    assert paths[0].edges[-1].to_node == "product p"