from src.models.cp_bandit import CPBandit
import numpy as np # type: ignore

def find_candidate(
        bandit: CPBandit,
        step_budget: int,
        cplist: list[int],
        delta_cand_step: float,
        sg_parameter: float,
        track_budget: bool = False
) -> tuple[int, bool] | tuple[int, bool, int]:
    """Find a candidate change point location.
    
    Args:
        bandit: The CPBandit instance
        step_budget: Budget for this step
        cplist: List of already detected change points
        delta_cand_step: Error probability for this step
        sg_parameter: SubGaussian parameter
        track_budget: Whether to track budget usage
        
    Returns:
        If track_budget=False: (candidate_location, success)
        If track_budget=True: (candidate_location, success, budget_used)
    """
    spent_budget = 0
    max_dist = 0
    compare_candidate = [0, 0]
    step_size = 1
    
    while step_size < bandit.num_arms - len(cplist) - 1:
        grid_size = int((bandit.num_arms - 1 - len(cplist)) / step_size)
        if 2 * grid_size * int(np.ceil(np.log2(grid_size))) <= step_budget:
            if track_budget:
                ba_estimate, budget_used = bandit.sequential_halving(
                    budget=step_budget,
                    cplist=cplist,
                    grid_dist=step_size,
                    track_budget=track_budget
                )
                spent_budget += budget_used
            else:
                ba_estimate = bandit.sequential_halving(
                    budget=step_budget,
                    cplist=cplist,
                    grid_dist=step_size
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
                    track_budget=True
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
    
    step_threshold = np.sqrt(4 * sg_parameter**2 * np.log(2 / delta_cand_step) / step_budget)
    
    if track_budget:
        test_result, used_budget = bandit.test_differences(
            step_budget,
            step_threshold,
            compare_candidate[0],
            compare_candidate[1],
            track_budget=True
        )
        spent_budget += used_budget
    else:
        test_result = bandit.test_differences(
            step_budget,
            step_threshold,
            compare_candidate[0],
            compare_candidate[1]
        )
    
    candidate = compare_candidate[0]
    
    if track_budget:
        return candidate, test_result, spent_budget
    else:
        return candidate, test_result


def localize_change_point(
        bandit: CPBandit,
        step_budget: int,
        candidate: int,
        cplist: list[int],
        delta_test_step: float,
        sg_parameter: float,
        track_budget: bool = False
) -> tuple[int, bool] | tuple[int, bool, int]:
    """Localize the change point from a candidate location.
    
    Args:
        bandit: The CPBandit instance
        step_budget: Budget for this step
        candidate: Candidate change point location
        cplist: List of already detected change points
        delta_test_step: Error probability for this step
        sg_parameter: SubGaussian parameter
        track_budget: Whether to track budget usage
        
    Returns:
        If track_budget=False: (changepoint_location, success)
        If track_budget=True: (changepoint_location, success, budget_used)
    """
    spent_budget = 0
    
    left = 0
    for i in cplist:
        if i < candidate:
            left = i
    
    if track_budget:
        cp_candidate, used_budget = bandit.bin_search(
            budget=step_budget,
            left=left,
            right=candidate,
            track_budget=True
        )
        spent_budget += used_budget
    else:
        cp_candidate = bandit.bin_search(
            budget=step_budget,
            left=left,
            right=candidate
        )
    
    step_threshold = np.sqrt(4 * sg_parameter**2 * np.log(2 / delta_test_step) / step_budget)
    
    if track_budget:
        test_result, used_budget = bandit.test_differences(
            step_budget,
            step_threshold,
            cp_candidate,
            track_budget=True
        )
        spent_budget += used_budget
    else:
        test_result = bandit.test_differences(
            step_budget,
            step_threshold,
            cp_candidate
        )
    
    if track_budget:
        return cp_candidate, test_result, spent_budget
    else:
        return cp_candidate, test_result


def fixed_confidence(
        bandit: CPBandit,
        min_cp_num: int,
        confidence: float = 0.05,
        track_budget: bool = False
) -> list[int] | tuple[list[int], int]:
    cplist = []
    spent_budget = 0
    step_budget = int(np.ceil(3*np.log2(bandit.num_arms)))
    sg_parameter = bandit.parameter
    candidate_detected = False
    delta_tilde = confidence / min_cp_num
    k = 1
    while len(cplist) < min_cp_num:
        if candidate_detected == False:
            delta_cand_step = delta_tilde / (2 * k**2)
            
            if track_budget:
                candidate, cand_success, used_budget = find_candidate(
                    bandit,
                    step_budget,
                    cplist,
                    delta_cand_step,
                    sg_parameter,
                    track_budget=True
                )
                spent_budget += used_budget
            else:
                candidate, cand_success = find_candidate(
                    bandit,
                    step_budget,
                    cplist,
                    delta_cand_step,
                    sg_parameter
                )
            
            if cand_success == True:
                candidate_detected = True
            else:
                step_budget *= 2
                k += 1
        else:
            delta_test_step = delta_tilde / (2 * k**2)
            
            if track_budget:
                cp_candidate, loc_success, used_budget = localize_change_point(
                    bandit,
                    step_budget,
                    candidate,
                    cplist,
                    delta_test_step,
                    sg_parameter,
                    track_budget=True
                )
                spent_budget += used_budget
            else:
                cp_candidate, loc_success = localize_change_point(
                    bandit,
                    step_budget,
                    candidate,
                    cplist,
                    delta_test_step,
                    sg_parameter
                )
            
            if loc_success == True:
                cplist.append(cp_candidate)
                cplist.sort()
                candidate_detected = False
                step_budget = int(np.ceil(3*np.log2(bandit.num_arms)))
                k = 1
            else:
                step_budget *= 2
                k += 1
                candidate_detected = False
    if track_budget:
        return cplist, spent_budget
    else:
        return cplist