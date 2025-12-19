
import numpy as np
from src.algorithms.sequential_halving_with_backtracking import shb 

def cmcpi(
        bandit,
        confidence: float,
        precision: float,
        num_cp: int
):
    tracked_budget = 0
    cplist = []
    iteration_budget = int(np.ceil(12*np.log(1/(2*precision))))
    k = 1
    
    while len(cplist) < num_cp:
        cp_detected = False
        
        while not cp_detected:
            depth = int(np.log2(1/precision)) + 1
            
            for l in range(depth):
                sect_budget = int(2**(-l) * iteration_budget / depth)
                
                if sect_budget < np.ceil(6*np.log(1/(2*precision))):
                    break
                    
                sect_length = 2**(-l)
                intervals = interval_partition(sect_length, cplist.copy(), precision)
                
                for interval in intervals:
                    cp_candidate, temp_budget = shb(
                        bandit,
                        sect_budget,
                        precision,
                        left=interval[0],
                        right=interval[1],
                        track_budget=True
                    )
                    tracked_budget += temp_budget
                    
                    mean_left = bandit.pull(max(cp_candidate - precision,0), sect_budget).mean()
                    mean_right = bandit.pull(min(cp_candidate + precision, 1.0), sect_budget).mean()
                    tracked_budget += 2 * sect_budget  
                    
                    threshold = np.sqrt(16*np.log((2*np.pi**2*num_cp*k**2)/(3*precision*confidence))/sect_budget)
                    
                    if abs(mean_left - mean_right) > threshold:
                        cplist.append(cp_candidate)
                        cplist.sort()
                        cp_detected = True
                        k = 1  
                        iteration_budget = int(np.ceil(12*np.log(1/(2*precision))))
                        break  
                
                if cp_detected:
                    break  
            
            if not cp_detected:
                k += 1
                iteration_budget = iteration_budget * 2
    
    return cplist, tracked_budget

         

def interval_partition(
        sect_length: float,
        cplist: list[float],
        precision: float
):
    intervals = []
    cplist.append(1+2*precision)
    next_cp = cplist.pop(0)
    left = 0.0
    right = sect_length
    while right <= 1.0:
        if right <= next_cp - precision:
            intervals.append([left, right])
            left += sect_length
            right += sect_length
        else:
            if left < next_cp - precision:
                intervals.append([left, next_cp - precision])
            left = next_cp + precision
            right = left + sect_length
            next_cp = cplist.pop(0)
    if left < 1.0:
        if next_cp <= 1.0:
            intervals.append([left, next_cp - precision])
            if next_cp + precision < 1.0:
                intervals.append([next_cp + precision, 1.0])
    return intervals

