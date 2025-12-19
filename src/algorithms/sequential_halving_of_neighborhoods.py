from src.models.continuous_cp_bandit import Continuous_Bandit
import numpy as np # type: ignore

def shon(
        bandit: Continuous_Bandit,
        budget: int,
        candidates: list[float],
        precision: float
) -> float:
    if 4 * len(candidates) > budget:
        raise ValueError("Budget too small for the number of candidates.")
    neighbors_list = neighbors(candidates, precision)
    num_halvings = int(np.floor(np.log2(len(candidates))))
    last_round_budget = budget // 4
    for i in range(num_halvings):
        step_budget = int(np.ceil(last_round_budget/ 2**(num_halvings - i - 1)))
        rewards = []
        for nb in neighbors_list:
            left, right = nb[0], nb[2] 
            rewards.append(abs(bandit.pull(right,step_budget).mean()-bandit.pull(left,step_budget)).mean())
        sorted_indices = np.argsort(rewards)[::-1]
        neighbors_list = [neighbors_list[idx] for idx in sorted_indices[:2**(num_halvings - i - 1)]]
    return neighbors_list[0][1]
        

def neighbors(
        candidates: list[float],
        precision: float
) -> list[list[float]]:
    neighbor_list = []
    for candidate in candidates:
        neighbor_list.append([max(candidate - precision, 0),candidate, min(candidate + precision, 1)])
    return neighbor_list