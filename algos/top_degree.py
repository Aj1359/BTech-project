import networkx as nx

def get_seeds(G, M, budget, **kwargs):
    vulnerable_set = kwargs.get('vulnerable_set', set(G.nodes()))
    candidates = list(vulnerable_set - set(M))
    degrees = dict(G.degree(candidates))
    sorted_candidates = sorted(degrees, key=degrees.get, reverse=True)
    return list(sorted_candidates[:budget])
