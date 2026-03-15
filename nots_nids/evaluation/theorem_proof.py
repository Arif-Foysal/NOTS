"""
evaluation/theorem_proof.py
===========================
Formal proof sketch for the NOTS detection floor guarantee (ε_min).

This module contains LaTeX-formatted documentation and helper functions to 
verify the mathematical properties required for the NOTS theorem.

Theorem: Nash-Optimal Detection Floor
-------------------------------------
Let D be the benign reference persistence diagram and D' be a live diagram
under L-∞ perturbation δ. If the projection P is L-Lipschitz, then:
    exp(DR) ≥ NashValue(U) - L * δ
where NashValue(U) is the value ε_min of the payoff matrix game.

Proof Sketch:
1. Stability (Cohen-Steiner): W_inf(D(X), D(X+Δ)) ≤ ||Δ||_∞.
2. Projection: If P is L-Lipschitz, then ||P(X) - P(X+Δ)||_∞ ≤ L * ||Δ||_∞.
3. PH Stability: W_inf(PH(P(X)), PH(P(X+Δ))) ≤ L * δ.
4. Minimax: By sampling from S*, the defender ensures a payoff of at least 
   ε_min against any strategy in the payoff matrix.
5. Bound: The empirical Wasserstein distance W for an attacked window is
   W ≥ W_benign - distortion.
"""

import numpy as np


def get_lipschitz_constant(method: str) -> float:
    """Return the theoretical Lipschitz constant for a projection method.
    
    References:
    - PCA: Orthogonal projections are non-expansive (L=1).
    - Random: JL-lemma ensures L ≈ 1 with high probability.
    - UMAP: Non-linear, L is technically unbounded (None).
    """
    if method == "pca":
        return 1.0
    elif method == "random":
        # For Gaussian Random Projection, L ≈ 1 with appropriate scaling
        return 1.0
    else:
        return np.inf


def verify_stability_bound(W_benign: float, W_attack: float, L: float, delta: float) -> bool:
    """Verify if the observed change in Wasserstein distance respects the stability bound.
    
    Bound: |W_attack - W_benign| ≤ L * delta
    """
    return abs(W_attack - W_benign) <= (L * delta + 1e-6)
