"""
DetectIntervals (Algorithm 1 in the paper): multiscale endpoint tests on dyadic grids.
"""

import numpy as np

def detect_intervals(bandit, delta, budget):
    """Detect disjoint intervals that each contain at least one change point.

    Args:
        bandit: Continuous_Bandit to sample from.
        delta: Confidence parameter.
        budget: Total sampling budget.

    Returns:
        List of detected intervals (left, right) and the budget spent.
    """
    intervals = []
    tracked_budget = 0
    dmax = int(np.log(budget/np.log(1/delta))/np.log(2))
    for d in range(dmax):
        num_intervals = 2**(d+1)
        step_budget = int(budget/(dmax*(num_intervals+1)))
        if step_budget == 0:
            break
        beta = np.sqrt((8/step_budget)*np.log(2*num_intervals*(dmax+1)/delta))
        xgrid = []
        ygrid = []
        for i in range(num_intervals+1):
            x = i/num_intervals
            xgrid.append(x)
            if step_budget > 1:
                ygrid.append(bandit.pull(x, step_budget).mean())
            else:
                ygrid.append(bandit.pull(x, step_budget))
            tracked_budget += step_budget
        for i in range(num_intervals):
            if abs(ygrid[i] - ygrid[i+1]) > beta:
                for interval in intervals[:]:
                    if interval[0] <= xgrid[i] and xgrid[i+1] <= interval[1]:
                        intervals.remove(interval)
                intervals.append((xgrid[i], xgrid[i+1]))
    return intervals, tracked_budget