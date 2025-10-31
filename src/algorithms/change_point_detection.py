from src.models.cp_bandit import CPBandit
import numpy as np # type: ignore

def fixed_confidence(
        bandit: CPBandit,
        min_cp_num: int,
        confidence: float = 0.05,
        track_budget: bool = False
) -> list[int] | tuple[list[int], int]:
    cplist = []
    spent_budget = 0
    step_budget = int(3*np.log2(bandit.num_arms))
    sg_parameter = bandit.parameter
    candidate_detected = False
    delta_candidate = confidence / (4 * min_cp_num )
    delfta_change_point = confidence / (4 * min_cp_num )
    while len(cplist) < min_cp_num:
        if candidate_detected == False:
            max_dist = 0
            step_size = 1
            while step_size < bandit.num_arms - len(cplist) - 1:
                grid_size = int((bandit.num_arms - 1 -len(cplist))/step_size)
                if 2* grid_size * int(np.ceil(np.log2(grid_size))) <= step_budget:
                    if track_budget:
                        ba_estimate, budget_used = bandit.sequential_halving(
                            budget = step_budget,
                            cplist=cplist,
                            grid_dist = step_size,
                            track_budget = track_budget
                        )
                        spent_budget += budget_used
                    else:
                        ba_estimate = bandit.sequential_halving(
                            budget = step_budget,
                            cplist=cplist,
                            grid_dist = step_size
                        )
                    i = 0
                    for j in cplist:
                        if j < ba_estimate:
                            i = j
                    if track_budget:
                        val, used_budget = bandit.calculate_mean_difference(
                            step_budget,
                            ba_estimate,
                            i,
                            track_budget = True
                        )
                        spent_budget += used_budget
                    else:
                        val = bandit.calculate_mean_difference(
                            step_budget,
                            ba_estimate,
                            i
                        )
                    if abs(val) > max_dist:
                        max_dist = abs(val)
                        compare_candidate = [ba_estimate, i]
                step_size *= 2
            step_threshold = np.sqrt( 4 *  sg_parameter**2 * np.log(  2/ delta_candidate ) / step_budget )
            if track_budget:
                test_result, used_budget = bandit.test_differences(
                    step_budget,
                    step_threshold,
                    compare_candidate[0],
                    compare_candidate[1],
                    track_budget = True
                )
                spent_budget += used_budget
            else:
                test_result = bandit.test_differences(
                    step_budget,
                    step_threshold,
                    compare_candidate[0],
                    compare_candidate[1]
                )
            if test_result == True :
                candidate = compare_candidate[0]
                candidate_detected = True
                delta_candidate = confidence / (4 * min_cp_num )
            else:
                step_budget *=2
                delta_candidate /= 2
            
            
        else:
            left = 0
            for i in cplist:
                if i < candidate:
                    left = i
            if track_budget:
                cp_candidate, used_budget = bandit.bin_search(
                    budget = step_budget,
                    left = left,
                    right = candidate,
                    track_budget = True
                )
                spent_budget += used_budget
            else:
                cp_candidate = bandit.bin_search(
                    budget = step_budget,
                    left = left,
                    right = candidate
                )
            step_threshold = np.sqrt( 4 *  sg_parameter**2 * np.log(  2/ delfta_change_point ) / step_budget )
            if track_budget:
                test_result, used_budget = bandit.test_differences(
                    step_budget,
                    step_threshold,
                    cp_candidate,
                    track_budget = True
                )
                spent_budget += used_budget
            else:
                test_result = bandit.test_differences(
                    step_budget,
                    step_threshold,
                    cp_candidate
                )
            if test_result == True:
                cplist.append(cp_candidate)
                cplist.sort()
                candidate_detected = False
                step_budget = int(3*np.log2(bandit.num_arms))
            else:
                delfta_change_point /= 2
                step_budget *= 2
    if track_budget:
        return cplist, spent_budget
    else:
        return cplist
    
        

