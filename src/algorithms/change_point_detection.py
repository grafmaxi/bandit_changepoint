from src.models.cp_bandit import CPBandit
import numpy as np # type: ignore

def fixed_confidence(
        bandit: CPBandit,
        min_cp_num: int,
        confidence: float = 0.05
) -> list[list[int], int]:
    cplist = []
    step_budget = int(3*np.log2(bandit.num_arms))
    while len(cplist) < min_cp_num:
        step_size = 1
        while step_size < bandit.num_arms - len(cplist) - 1:
            grid_size = int((bandit.num_arms - 1 -len(cplist))/step_size)
            print(grid_size, step_budget)
            if 2* grid_size * int(np.ceil(np.log2(grid_size))) <= step_budget:
                ba_estimate = bandit.sequential_halving(
                    budget = step_budget,
                    cplist=cplist,
                    grid_dist = step_size
                )
                i = 0
                for j in cplist:
                    if j < ba_estimate:
                        i = j
                if bandit.test_differences(
                    step_budget,
                    0.95,
                    i,
                    ba_estimate
                ) == True:
                    cplist.append(ba_estimate)
                    cplist.sort()
                    break
            step_size *= 2
        step_budget *= 2
    return [cplist, step_budget]
    
        
