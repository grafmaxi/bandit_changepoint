import numpy as np
from src.models.continuous_cp_bandit import Continuous_Bandit
from src.algorithms.continuous_multiple_change_point_identification import cmcpi

def test_change_point_detection():
    # Define test scenarios
    precision = 0.1
    confidence = 0.1
    cplist = [0.2, 0.5, 0.7]
    values = [0.2, 0.8, 0.3, 0.9]
    bandit = Continuous_Bandit(change_points=cplist, values=values, sampling_type="gaussian", parameter=1)
    num_cp = len(cplist)   

    detected_cplist, tracked_budget = cmcpi(bandit, confidence, precision, num_cp)
    return detected_cplist, tracked_budget

if __name__ == "__main__":
    num_tests = 100
    for i in range(num_tests):
        np.random.seed(i)
        detected_cplist, tracked_budget = test_change_point_detection()
        print(f"Test {i+1}: Detected Change Points: {detected_cplist}, Tracked Budget: {tracked_budget}")   
