import numpy as np # type: ignore

def estimate_jumps(bandit, intervals, delta, budget, num_cp):
    estimates = {}
    good_estimates = {}
    tracked_budget = 0
    k = 1
    for interval in intervals:
        estimates[interval] = 0
    while len(good_estimates) < num_cp and tracked_budget + max((len(intervals) - len(good_estimates)), 1) * 2**k < budget and len(estimates) > 0:
        beta = np.sqrt(8*np.log(np.pi**2*len(intervals)*k**2/(3*delta))/2**k)
        for interval in list(estimates):
            if k == 1:
                estimates[interval] = abs(bandit.pull(interval[0]) - bandit.pull(interval[1]))
            else:
                estimates[interval] = abs(bandit.pull(interval[0], 2**(k-1)).mean() - bandit.pull(interval[1], 2**(k-1)).mean())
            tracked_budget += 2**k
            if estimates[interval] > beta:
                good_estimates[interval] = estimates.pop(interval)
        k += 1
    return good_estimates, tracked_budget
            
           
    
    
