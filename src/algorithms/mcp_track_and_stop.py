"""
MCPI: Track-and-Stop for multiple change points on a discrete bandit (Lazzaro and Pike-Burke, 2025a).
"""

from src.models.cp_bandit import CPBandit
import numpy as np
import math


def mcp_track_and_stop(bandit: CPBandit, delta: float, num_find: int = 1, update: bool = False):
    """Identify num_find change points of a discrete change point bandit.

    Args:
        bandit: CPBandit to sample from.
        delta: Confidence parameter.
        num_find: Number of change points to identify.
        update: Use the update condition (19) of Lazzaro and Pike-Burke (2025a) for the candidate.

    Returns:
        Indices of the identified change points (last arm before each jump) and the number of pulls.
    """
    num_arms = bandit.num_arms
    sum_vec = np.zeros(num_arms)
    mean_vec = np.zeros(num_arms)
    num_vec = np.zeros(num_arms)
    diff_vec = np.zeros(num_arms - 1)
    # Differences at already identified change points are no longer tracked
    mask = np.ones(num_arms-1, dtype=bool)
    cps = []
    t = 0
    # Sample every arm once
    for i in range(num_arms):
        sum_vec[i] = bandit.pull(i)
        mean_vec[i] = sum_vec[i]
        num_vec[i] = 1
        if i > 0:
            diff_vec[i-1] = abs(mean_vec[i] - mean_vec[i-1])
        t += 1
    forced_exploration = 0
    # Stopping threshold beta
    gamma = 2 * np.exp(3) * 9 **6 / np.log(3)
    arg = t * gamma * (num_arms - 1) * num_find / delta
    log_arg = math.log(arg)
    beta = log_arg + 8 * math.log(log_arg)
    # Cycle through the arms during the initial forced exploration; only speeds up the code
    still_exploring = True
    # Threshold r of the update condition
    if update:
        r = np.sqrt( (4 * np.log(t) + 2* np.log( 2 * np.log(t)) + 0.5) / max(t**(0.25) - num_arms, np.finfo(np.float32).eps) )
    for j in range(num_find):
        xhat = np.argmax(diff_vec)
        # Test statistic for a jump between arms xhat and xhat + 1
        z = (num_vec[xhat] * num_vec[xhat+1]) / (2*(num_vec[xhat] + num_vec[xhat+1])) * (mean_vec[xhat] - mean_vec[xhat+1])**2
        while z < beta:
            # Update condition (19) in Lazzaro and Pike-Burke (2025a)
            if update:
                if max(diff_vec[mask]) > r + min(diff_vec[mask]):
                    xhat = np.argmax(diff_vec)
            # Forced exploration: every arm is sampled at least sqrt(t) times
            if num_vec.min() < t**0.5:
                # Sample the arm with the least number of pulls
                if still_exploring:
                    a = forced_exploration
                    forced_exploration = (forced_exploration + 1) % num_arms
                else:
                    a = np.argmin(num_vec)
                sum_vec[a] += bandit.pull(a)
                num_vec[a] += 1
                mean_vec[a] = sum_vec[a] / num_vec[a]
                if a < num_arms - 1 and mask[a]:
                    diff_vec[a] = abs(mean_vec[a] - mean_vec[a+1])
                    if (not update) and diff_vec[a] > diff_vec[xhat]:
                        xhat = a
                if a > 0 and mask[a-1]:
                    diff_vec[a-1] = abs(mean_vec[a] - mean_vec[a-1])
                    if (not update) and diff_vec[a-1] > diff_vec[xhat]:
                        xhat = a-1
                t += 1
            else:
                still_exploring = False
                # Sample the less-pulled of the two arms next to the candidate xhat
                if num_vec[xhat] <= num_vec[xhat+1]:
                    a = xhat
                else:
                    a = xhat + 1
                sum_vec[a] += bandit.pull(a)
                num_vec[a] += 1
                mean_vec[a] = sum_vec[a] / num_vec[a]
                if a < num_arms - 1 and mask[a]:
                    diff_vec[a] = abs(mean_vec[a] - mean_vec[a+1])
                    if (not update) and diff_vec[a] > diff_vec[xhat]:
                        xhat = np.argmax(diff_vec)
                if a > 0 and mask[a-1]:
                    diff_vec[a-1] = abs(mean_vec[a] - mean_vec[a-1])
                    if (not update) and diff_vec[a-1] > diff_vec[xhat]:
                        xhat = np.argmax(diff_vec)
                t += 1
            z = (num_vec[xhat] * num_vec[xhat+1]) / (2*(num_vec[xhat] + num_vec[xhat+1])) * (mean_vec[xhat] - mean_vec[xhat+1])**2
            arg = t * gamma * (num_arms - 1) * num_find / delta
            log_arg = math.log(arg)
            beta = log_arg + 8 * math.log(log_arg)
            if update:
                r = np.sqrt( (4 * np.log(t) + 2* np.log( 2 * np.log(t)) + 0.5) / max(t**(0.25) - num_arms, np.finfo(np.float32).eps) )
        # z >= beta: accept xhat and exclude it from the following rounds
        cps.append(xhat)
        mask[xhat] = False  
        diff_vec[xhat] = -np.inf
    return cps, t  