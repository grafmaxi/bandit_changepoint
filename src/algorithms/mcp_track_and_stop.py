from src.models.cp_bandit import CPBandit
import numpy as np # type: ignore





def mcp_track_and_stop(bandit: CPBandit, delta: float, num_find: int = 1, update: bool = False):
    arms = bandit.arms
    num_arms = bandit.num_arms
    sum_vec = np.zeros(num_arms)
    mean_vec = np.zeros(num_arms)
    num_vec = np.zeros(num_arms)
    diff_vec = np.zeros(num_arms - 1)
    mask = np.ones(num_arms-1, dtype=bool)
    cps = []
    t = 0
    for i in range(num_arms):
        sum_vec[i] = bandit.pull(i)
        mean_vec[i] = sum_vec[i]
        num_vec[i] = 1
        if i > 0:
            diff_vec[i-1] = abs(mean_vec[i] - mean_vec[i-1])
        t += 1
    gamma = 2 * np.exp(3) * 9 **6 / np.log(3)
    beta = np.log( t* gamma * (num_arms-1) * num_find / delta ) + 8 * np.log(np.log(t*gamma*(num_arms-1)*num_find/delta))
    r = np.sqrt( (4 * np.log(t) + 2* np.log( 2 * np.log(t)) + 0.5) / max(t**(0.25) - num_arms, np.finfo(np.float32).eps) )
    for j in range(num_find):
        xhat = np.argmax(diff_vec)
        z = (num_vec[xhat] * num_vec[xhat+1]) / (2*(num_vec[xhat] + num_vec[xhat+1])) * (mean_vec[xhat] - mean_vec[xhat+1])**2
        while z < beta:
            if update:
                if max(diff_vec[mask]) > r + min(diff_vec[mask]):
                    xhat = np.argmax(diff_vec)
            else:
                xhat = np.argmax(diff_vec)
            if min(num_vec) < np.sqrt(t):
                a = np.argmin(num_vec)
                sum_vec[a] += bandit.pull(a)
                num_vec[a] += 1
                mean_vec[a] = sum_vec[a] / num_vec[a]
                if a < num_arms - 1 and mask[a]:
                    diff_vec[a] = abs(mean_vec[a] - mean_vec[a+1])
                if a > 0 and mask[a-1]:
                    diff_vec[a-1] = abs(mean_vec[a] - mean_vec[a-1])
                t += 1
            else:
                if num_vec[xhat] <= num_vec[xhat+1]:
                    a = xhat
                else:
                    a = xhat + 1
                sum_vec[a] += bandit.pull(a)
                num_vec[a] += 1
                mean_vec[a] = sum_vec[a] / num_vec[a]
                if a < num_arms - 1 and mask[a]:
                    diff_vec[a] = abs(mean_vec[a] - mean_vec[a+1])
                if a > 0 and mask[a-1]:
                    diff_vec[a-1] = abs(mean_vec[a] - mean_vec[a-1])
                t += 1
            z = (num_vec[xhat] * num_vec[xhat+1]) / (2*(num_vec[xhat] + num_vec[xhat+1])) * (mean_vec[xhat] - mean_vec[xhat+1])**2
            beta = np.log( t* gamma * (num_arms-1) * num_find / delta ) + 8 * np.log(np.log(t*gamma*(num_arms-1)*num_find/delta))
            r = np.sqrt( (4 * np.log(t) + 2* np.log( 2 * np.log(t)) + 0.5) / max(t**(0.25) - num_arms, np.finfo(np.float32).eps) )
        cps.append(xhat)
        mask[xhat] = False  
        diff_vec[xhat] = -np.inf
    return cps, t  