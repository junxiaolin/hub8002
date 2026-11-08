"""Baseline hub rules. Each returns a sorted list of player indices (into k / T / nodes)."""
import networkx as nx
import numpy as np

from .mdl import mdl_hubs

METHODS = ('count_top', 'per90_top', 'pagerank_top', 'mean2sd', 'average', 'loubar',
           'mdl_cm', 'mdl_er', 'mdl_cm_min', 'mdl_er_min')


def _std_counts(k, T):
    """Counts rescaled to equal exposure (mean T) and rounded, as in the pilot."""
    k, T = np.asarray(k, float), np.asarray(T, float)
    return np.rint(k * T.mean() / np.where(T > 0, T, np.nan)).astype(int)


def pagerank_scores(nodes, edges, weight='w', alpha=0.85):
    """PageRank on the directed pass graph i -> j (weight = passes); `edges` iterable of (i, j, w)."""
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for i, j, w in edges:
        if w > 0:
            G.add_edge(i, j, **{weight: float(w)})
    if G.number_of_edges() == 0:
        return {v: 1.0 / max(len(nodes), 1) for v in nodes}
    return nx.pagerank(G, alpha=alpha, weight=weight)


def baseline_hubs(method, k=None, T=None, pr=None):
    """k: counts, T: exposure (90-min units), pr: PageRank scores aligned with k."""
    k = None if k is None else np.asarray(k, float)
    if method == 'count_top':
        return [int(np.argmax(k))]
    if method == 'per90_top':
        return [int(np.argmax(k / np.asarray(T, float)))]
    if method == 'pagerank_top':
        return [int(np.argmax(pr))]
    if method == 'mean2sd':
        return [int(i) for i in np.flatnonzero(k > k.mean() + 2 * k.std(ddof=1))]
    if method == 'average':
        return [int(i) for i in np.flatnonzero(k >= k.mean())]
    if method == 'loubar':
        thr = np.percentile(k, (1 - k.mean() / k.max()) * 100) if k.max() > 0 else np.inf
        return [int(i) for i in np.flatnonzero(k >= thr)]
    if method in ('mdl_cm', 'mdl_er'):
        return mdl_hubs(np.rint(k).astype(int))[method[-2:].upper()][0]
    if method in ('mdl_cm_min', 'mdl_er_min'):
        return mdl_hubs(_std_counts(k, T))[method[4:6].upper()][0]
    raise ValueError(method)
