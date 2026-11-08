import numpy as np

from hub8002.models.mdl import mdl_hubs, official_hubs


def test_port_matches_official_on_random_degree_sequences():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(5, 30))
        deg = rng.negative_binomial(2, 0.02, n)
        deg[rng.integers(0, n)] += int(rng.integers(0, 400))
        port = mdl_hubs(deg)
        off = official_hubs(deg, weighted=True)
        assert port['ER'][0] == off['ER']
        assert port['CM'][0] == off['CM']
