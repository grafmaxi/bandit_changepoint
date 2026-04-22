from src.models.cp_bandit import CPBandit
import numpy as np # type: ignore





def mcp_track_and_stop(bandit: CPBandit, delta: float, num_find: int = 1, update: bool = False):
    num_arms = bandit.num_arms
    # Initialize vectors to store sums, means, counts, and differences
    sum_vec = np.zeros(num_arms)
    mean_vec = np.zeros(num_arms)
    num_vec = np.zeros(num_arms)
    diff_vec = np.zeros(num_arms - 1)
    # Initialize mask to indicate which differences we want to track (initially all True)
    mask = np.ones(num_arms-1, dtype=bool)
    # List to store identified change points
    cps = []
    # Budget counter
    t = 0
    # Initial sampling of each arm
    for i in range(num_arms):
        sum_vec[i] = bandit.pull(i)
        mean_vec[i] = sum_vec[i]
        num_vec[i] = 1
        if i > 0:
            diff_vec[i-1] = abs(mean_vec[i] - mean_vec[i-1])
        t += 1
    # Calculate the initial stopping threshold parameters
    gamma = 2 * np.exp(3) * 9 **6 / np.log(3)
    beta = np.log( t* gamma * (num_arms-1) * num_find / delta ) + 8 * np.log(np.log(t*gamma*(num_arms-1)*num_find/delta))
    # Calculate the initial parameter for the update condition
    if update:
        r = np.sqrt( (4 * np.log(t) + 2* np.log( 2 * np.log(t)) + 0.5) / max(t**(0.25) - num_arms, np.finfo(np.float32).eps) )
    # Main loop to identify change points
    for j in range(num_find):
        # Find the index of the maximum difference
        xhat = np.argmax(diff_vec)
        # Calculate the test statistic z for xhat 
        z = (num_vec[xhat] * num_vec[xhat+1]) / (2*(num_vec[xhat] + num_vec[xhat+1])) * (mean_vec[xhat] - mean_vec[xhat+1])**2
        # While the test statistic z is below the threshold beta, continue sampling
        while z < beta:
            # Corresponds to condition (19) in the paper of Lazzaro & Pike-Burke 
            if update:
                if max(diff_vec[mask]) > r + min(diff_vec[mask]):
                    xhat = np.argmax(diff_vec)
            else:
                # If the update condition is not met, simply update xhat to be the index of the maximum difference
                xhat = np.argmax(diff_vec)
            # Guatantees that each arm is sampled at least sqrt(t) times 
            if min(num_vec) < np.sqrt(t):
                # Sample the arm with the least number of pulls
                a = np.argmin(num_vec)
                # Update the sum, count, mean and differences for the selected arm
                sum_vec[a] += bandit.pull(a)
                num_vec[a] += 1
                mean_vec[a] = sum_vec[a] / num_vec[a]
                if a < num_arms - 1 and mask[a]:
                    diff_vec[a] = abs(mean_vec[a] - mean_vec[a+1])
                if a > 0 and mask[a-1]:
                    diff_vec[a-1] = abs(mean_vec[a] - mean_vec[a-1])
                t += 1
            else:
                # Sample the arm with the least number of pulls among the two arms corresponding to xhat
                if num_vec[xhat] <= num_vec[xhat+1]:
                    a = xhat
                else:
                    a = xhat + 1
                # Update the sum, count, mean and differences for the selected arm
                sum_vec[a] += bandit.pull(a)
                num_vec[a] += 1
                mean_vec[a] = sum_vec[a] / num_vec[a]
                if a < num_arms - 1 and mask[a]:
                    diff_vec[a] = abs(mean_vec[a] - mean_vec[a+1])
                if a > 0 and mask[a-1]:
                    diff_vec[a-1] = abs(mean_vec[a] - mean_vec[a-1])
                t += 1
            # Update the test statistic z, the threshold beta and the parameter r for the update condition
            z = (num_vec[xhat] * num_vec[xhat+1]) / (2*(num_vec[xhat] + num_vec[xhat+1])) * (mean_vec[xhat] - mean_vec[xhat+1])**2
            beta = np.log( t* gamma * (num_arms-1) * num_find / delta ) + 8 * np.log(np.log(t*gamma*(num_arms-1)*num_find/delta))
            if update:
                r = np.sqrt( (4 * np.log(t) + 2* np.log( 2 * np.log(t)) + 0.5) / max(t**(0.25) - num_arms, np.finfo(np.float32).eps) )
        # Once the test statistic z exceeds the threshold beta, we have identified a change point at index xhat
        cps.append(xhat)
        mask[xhat] = False  
        # Set the difference at xhat to -inf to ensure it is not selected again
        diff_vec[xhat] = -np.inf
    return cps, t  