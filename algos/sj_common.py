"""
sj_common.py -- Shared spreadability calculations and LFPR power iterations.
"""
import numpy as np
import scipy.sparse as sps


def spreadability(A_out, rho_eff, pT, undirected=False):
    """
    Computes 2-hop spreadability:
      kappa(u) = sum_{v within 2 hops, v != u} pT^dist(u,v) * rho_eff(v)
    where dist(u, v) in {1, 2} is the exact shortest path distance in G.
    """
    n = A_out.shape[0]
    pT = float(pT)
    rho_eff = np.asarray(rho_eff, dtype=float)

    # 1-hop adjacency (binary)
    A1 = sps.csr_matrix(A_out > 0, dtype=float)

    # 1-hop contribution
    # kappa_1[u] = pT * sum_{v in N(u)} rho_eff(v)
    kappa_1 = pT * (A1 @ rho_eff)

    # 2-hop paths: A2 = A1 @ A1
    A2 = A1 @ A1

    # Remove self (diagonal) and existing 1-hop neighbors
    A2.setdiag(0.0)
    A2.eliminate_zeros()

    # Mask out edges that already exist in A1
    # For sparse matrices: A2_exact = (A2 > 0) - (A1 > 0)
    A2_bin = sps.csr_matrix(A2 > 0, dtype=float)
    A2_exact = A2_bin - A1.multiply(A2_bin)
    A2_exact.eliminate_zeros()

    # 2-hop contribution
    kappa_2 = (pT ** 2) * (A2_exact @ rho_eff)

    kappa = kappa_1 + kappa_2
    return np.asarray(kappa, dtype=float).ravel()


def _lfpr_walk(W, WT, v, gP, nu, phi_walk, iters=300, tol=1e-9):
    """
    Locally Fair PageRank walk with weighted transitions W (W[j, i] = a_ji).
    Every node j allocates a fraction phi_walk of its walk mass to protected nodes (gP),
    and (1 - phi_walk) to unprotected nodes (gU = 1 - gP), proportional to W[j, i].
    
    If out_g(j) == 0 (no outgoing neighbor in group g), that group's fraction
    is redistributed uniformly over that group.
    
    Solve:
      pi = nu * v + (1 - nu) * P^T pi
    """
    n = W.shape[0]
    gP = np.asarray(gP, dtype=float).ravel()
    gU = 1.0 - gP
    nP = float(gP.sum())
    nU = float(gU.sum())

    if nP == 0.0 or nU == 0.0:
        phi_walk = 1.0 if nU == 0.0 else 0.0

    # Total outgoing edge weights to P and U from each node j:
    # outP[j] = sum_{i in P} W[j, i]
    # outU[j] = sum_{i in U} W[j, i]
    outP = np.asarray(W @ gP).ravel()
    outU = np.asarray(W @ gU).ravel()

    invP = np.where(outP > 0, 1.0 / np.maximum(outP, 1e-12), 0.0)
    invU = np.where(outU > 0, 1.0 / np.maximum(outU, 1e-12), 0.0)

    danP = outP == 0.0
    danU = outU == 0.0

    p = v.copy()
    if p.sum() > 0:
        p /= p.sum()
    else:
        p = np.full(n, 1.0 / max(1, n))

    for _ in range(iters):
        # Contribution from normal transitions:
        # For each target i: sum_j p[j] * [ phi * W[j,i]/outP[j] * I(i in P) + (1-phi) * W[j,i]/outU[j] * I(i in U) ]
        # WT is W.T, so (WT @ (p * invP))[i] = sum_j W[j, i] * p[j] / outP[j]
        flowP = WT @ (p * invP)
        flowU = WT @ (p * invU)

        # Contribution from dangling nodes:
        dan_massP = float((p * danP).sum()) / max(nP, 1.0)
        dan_massU = float((p * danU).sum()) / max(nU, 1.0)

        # Combine protected targets and unprotected targets
        prop_step = phi_walk * gP * (flowP + dan_massP) + (1.0 - phi_walk) * gU * (flowU + dan_massU)

        p_next = nu * v + (1.0 - nu) * prop_step
        diff = float(np.abs(p_next - p).sum())
        p = p_next
        if diff < tol:
            break

    s = p.sum()
    if s > 0:
        p /= s
    return p
