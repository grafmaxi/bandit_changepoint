"""
LCP (LocalizeChangePoints, Algorithm 2 in the paper) and VerifyCP (Algorithm 5).
"""

from src.algorithms.detect_intervals import detect_intervals
from src.algorithms.estimate_jumps import estimate_jumps
from src.algorithms.sequential_halving_with_backtracking import shb
import numpy as np

def localize_change_points(bandit, num_cps, delta, delta_explore, precision):
    """Localize num_cps change points to the given precision with confidence 1 - delta.

    Args:
        bandit: Continuous_Bandit to sample from.
        num_cps: Number N of change points to localize.
        delta: Confidence parameter for correctness.
        delta_explore: Confidence parameter of the exploration phase, controls the budget.
        precision: Target precision eta.

    Returns:
        Sorted estimated change points, the total budget, and the budgets of the
        detection, estimation, localization and verification phases.
    """
    cps = []
    k = int(np.ceil(np.log2(num_cps)))
    tracked_budget = 0
    tracked_detection_budget = 0
    tracked_estimation_budget = 0
    tracked_localization_budget = 0
    tracked_verification_budget = 0
    while True:
        intervals, step_budget = detect_intervals(bandit, delta_explore/4, 2**k)
        tracked_budget += step_budget
        tracked_detection_budget += step_budget
        if len(intervals) >= num_cps:
            estimates, step_budget = estimate_jumps(bandit, intervals, delta_explore/4, 2**k, num_cps)
            tracked_budget += step_budget
            tracked_estimation_budget += step_budget
            if len(estimates) >= num_cps:
                s = np.sum(np.array(list(estimates.values()))**(-2))
                delta_verify = 3*delta/(2*np.pi**2*k**2*num_cps)
                verify = True
                for v in range(num_cps):
                    interval = list(estimates)[v]
                    t_step = max(int((estimates[interval]**(-2) / s) * 2**k),2)
                    cp, step_budget = shb(bandit, t_step, precision, interval, track_budget=True)
                    cps.append(cp)
                    tracked_budget += step_budget
                    tracked_localization_budget += step_budget
                    if interval[1] - interval[0] > 2 * precision:
                        left = max(interval[0], cp - precision)
                        right = min(interval[1], cp + precision)
                        verify, step_budget = verify_change_point(bandit, left, right, delta_verify, t_step)
                        tracked_budget += step_budget
                        tracked_verification_budget += step_budget
                        if not verify:
                            break
                if verify:
                    return np.sort(cps), tracked_budget, tracked_detection_budget, tracked_estimation_budget, tracked_localization_budget, tracked_verification_budget
        cps = []
        k += 1



                        


def verify_change_point(bandit, left, right, delta, budget):
    """Two-sample test for a jump between the points left and right.

    Args:
        bandit: Continuous_Bandit to sample from.
        left: Left test point.
        right: Right test point.
        delta: Confidence parameter.
        budget: Total sampling budget, split evenly between both points.

    Returns:
        Whether a jump was detected, and the budget spent.
    """
    if budget < 2:
        return False, 0
    beta = np.sqrt(32*np.log(2/delta)/budget)
    tau = int(budget/2)
    if tau == 1:
        estimate_jump = abs(bandit.pull(left) - bandit.pull(right))
        return estimate_jump > beta, 2 * tau
    else: 
        estimate_jump = abs(bandit.pull(left, tau).mean() - bandit.pull(right, tau).mean())
        return estimate_jump > beta, 2 * tau
