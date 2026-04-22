from src.models.cp_bandit import CPBandit
import numpy as np # type: ignore





def scp_track_and_stop(bandit: CPBandit, delta: float):
    arms = bandit.arms
    num_arms = bandit.num_arms
    sum_vec = np.zeros(num_arms)
    mean_vec = np.zeros(num_arms)
    num_vec = np.zeros(num_arms)
    diff_vec = np.zeros(num_arms - 1)
    t = 0
    for i in range(num_arms):
        sum_vec[i] = bandit.pull(i)
        mean_vec[i] = sum_vec[i]
        num_vec[i] = 1
        if i > 0:
            diff_vec[i-1] = abs(mean_vec[i] - mean_vec[i-1])
        t += 1
    gamma = 2 * np.exp(3) * 9 **6 / np.log(3)
    beta = np.log( t* gamma * (num_arms-1) / delta ) + 8 * np.log(np.log(t*gamma*(num_arms-1)/delta))
    mean_diff = abs(mean_vec[0]-mean_vec[1])
    xhat = np.argmax(diff_vec)
    z = (num_vec[xhat] * num_vec[xhat+1]) / (2*(num_vec[xhat] + num_vec[xhat+1])) * (mean_vec[xhat] - mean_vec[xhat+1])**2
    while z < beta:
        if min(num_vec) < np.sqrt(t):
            a = np.argmin(num_vec)
            sum_vec[a] += bandit.pull(a)
            num_vec[a] += 1
            mean_vec[a] = sum_vec[a] / num_vec[a]
            if a== 0:
                diff_vec[a] = abs(mean_vec[a] - mean_vec[a+1])
            elif a == num_arms - 1:
                diff_vec[a-1] = abs(mean_vec[a] - mean_vec[a-1])
            else:
                diff_vec[a] = abs(mean_vec[a] - mean_vec[a+1])
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
            if a== 0:
                diff_vec[a] = abs(mean_vec[a] - mean_vec[a+1])
            elif a == num_arms - 1:
                diff_vec[a-1] = abs(mean_vec[a] - mean_vec[a-1])
            else:
                diff_vec[a] = abs(mean_vec[a] - mean_vec[a+1])
                diff_vec[a-1] = abs(mean_vec[a] - mean_vec[a-1])
            t += 1
        xhat = np.argmax(diff_vec)
        z = (num_vec[xhat] * num_vec[xhat+1]) / (2*(num_vec[xhat] + num_vec[xhat+1])) * (mean_vec[xhat] - mean_vec[xhat+1])**2
        beta = np.log( t* gamma * (num_arms-1) / delta ) + 8 * np.log(np.log(t*gamma*(num_arms-1)/delta))
    return xhat, t  # return estimated change point and total pulls used