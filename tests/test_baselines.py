import networkx as nx
import numpy as np

from hub8002.models.baselines import baseline_hubs, pagerank_scores


def test_simple_rules_on_toy_counts():
    k = np.array([10, 20, 30, 100])
    T = np.array([1.0, 1.0, 1.0, 10.0])
    assert baseline_hubs('count_top', k, T) == [3]
    assert baseline_hubs('per90_top', k, T) == [2]
    assert baseline_hubs('average', k, T) == [3]           # mean 40
    k2 = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 100])
    assert baseline_hubs('mean2sd', k2, np.ones(10)) == [9]
    thr = np.percentile(k, (1 - k.mean() / k.max()) * 100)
    assert baseline_hubs('loubar', k, T) == [int(i) for i in np.flatnonzero(k >= thr)]


def test_pagerank_matches_networkx():
    edges = [(0, 1, 5), (1, 2, 3), (2, 0, 1), (2, 1, 4)]
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges, weight='w')
    ref = nx.pagerank(G, weight='w')
    pr = pagerank_scores([0, 1, 2], edges)
    assert all(abs(pr[v] - ref[v]) < 1e-12 for v in ref)
    assert baseline_hubs('pagerank_top', pr=[pr[v] for v in range(3)]) == [1]
