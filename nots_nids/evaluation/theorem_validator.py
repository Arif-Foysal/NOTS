"""
evaluation/theorem_validator.py
================================
Validate the ε_min bound — the most important scientific check in the paper.

For each δ, check that empirical DR ≥ ε_min.  A violation means the theorem
proof or the implementation has a gap.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def validate_epsilon_min_bound(
    exp2_results: Dict[float, Dict[str, float]],
    epsilon_min: float,
    tolerance: float = 0.02,
) -> Dict[str, Any]:
    """Validate the ε_min theorem bound against empirical results.

    Parameters
    ----------
    exp2_results : dict
        ``{delta: {'DR': float, ...}}`` — empirical detection rates from Exp 2.
    epsilon_min : float
        The Nash-equilibrium-guaranteed detection floor.
    tolerance : float
        Numerical noise tolerance (default 2 %).

    Returns
    -------
    verdict : dict
        ``{'bound_holds': bool,
           'violations': list[(delta, DR, epsilon_min)],
           'margin': float,
           'max_delta_holding': float or None,
           'summary': str}``
    """
    violations: List[Tuple[float, float, float]] = []
    margins: List[float] = []
    max_delta_holding: float = 0.0

    for delta in sorted(exp2_results.keys()):
        empirical_dr = exp2_results[delta]["DR"]
        margin = empirical_dr - epsilon_min
        margins.append(margin)

        if empirical_dr < epsilon_min - tolerance:
            violations.append((delta, empirical_dr, epsilon_min))
            logging.critical(
                "THEOREM BOUND VIOLATED at delta=%.4f: "
                "empirical DR=%.4f < epsilon_min=%.4f. "
                "Review proof or implementation.",
                delta,
                empirical_dr,
                epsilon_min,
            )
        else:
            max_delta_holding = delta

    bound_holds = len(violations) == 0
    min_margin = min(margins) if margins else 0.0

    if bound_holds:
        summary = (
            f"✓ Theorem bound HOLDS for all δ values. "
            f"ε_min={epsilon_min:.4f}, minimum margin={min_margin:.4f}. "
            f"Bound valid up to δ={max_delta_holding:.4f}."
        )
        logger.info(summary)
    else:
        summary = (
            f"✗ THEOREM BOUND VIOLATED at {len(violations)} δ value(s). "
            f"ε_min={epsilon_min:.4f}. "
            f"Violations: {[(d, f'DR={dr:.4f}') for d, dr, _ in violations]}. "
            f"Review proof assumptions or implementation."
        )
        logger.critical(summary)

    return {
        "bound_holds": bound_holds,
        "violations": violations,
        "margin": min_margin,
        "max_delta_holding": max_delta_holding if bound_holds else None,
        "summary": summary,
    }
