"""
CSCPI: earlier single change point method that repeats SHB with a verification test and doubling budget.
"""

from src.algorithms.sequential_halving_with_backtracking import shb
import numpy as np

def cscpi(
        bandit,
        confidence: float,
        precision: float
):
    """Localize a single change point to the given precision with the given confidence.

    Args:
        bandit: Continuous_Bandit to sample from.
        confidence: Confidence parameter.
        precision: Target precision eta.

    Returns:
        Estimated change point and the budget spent.
    """
    tracked_budget = 0
    cpfound = False
    iteration_budget = int(60*np.log(1/(2*precision)))
    k = 1
    tracked_budget = 0
    
    while not cpfound:
        cp, step_budget = shb(
            bandit,
            iteration_budget,
            precision,
            track_budget=True
        )
        tracked_budget += step_budget
        mean_left = bandit.pull(max(cp - precision,0), iteration_budget).mean()
        mean_right = bandit.pull(min(cp + precision, 1.0), iteration_budget).mean()
        tracked_budget += 2 * iteration_budget  
        if abs(mean_left - mean_right) > np.sqrt(4*np.log(7*k**2/confidence)/iteration_budget):
            cpfound = True 
        else:
            k += 1
            iteration_budget *= 2
    return cp, tracked_budget