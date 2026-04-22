from src.algorithms.continuous_single_change_point_identification import cscpi
from src.models.continuous_cp_bandit import Continuous_Bandit
from src.models.cp_bandit import CPBandit
from src.algorithms.scp_track_and_stop import scp_track_and_stop
import numpy as np # type: ignore

def test_single_change_point_identification_continuous():
    # Define test scenario
    precision = 0.001
    confidence = 0.1
    cplist = [0.4]
    values = [0, 0.5]
    bandit = Continuous_Bandit(change_points=cplist, values=values, sampling_type="gaussian", parameter=1)
    
    detected_cp, tracked_budget = cscpi(bandit, confidence, precision)
    return detected_cp, tracked_budget  

if __name__ == "__main__":
    num_tests = 10
    for i in range(num_tests):
        np.random.seed(i)
        detected_cp, tracked_budget = test_single_change_point_identification_continuous()
        print(f"Test {i+1}: Detected Change Point: {detected_cp}, Tracked Budget: {tracked_budget}")

def test_single_change_point_identification_discrete():
    # Define test scenario
    delta = 0.1
    sec_len = [200, 300]
    sec_means = [0, .5]
    bandit = CPBandit(sec_len=sec_len, sec_means=sec_means, sampling_type="gaussian", parameter=1)
    
    detected_cp, total_pulls = scp_track_and_stop(bandit, delta)
    return detected_cp, total_pulls

if __name__ == "__main__":
    num_tests = 10
    for i in range(num_tests):
        np.random.seed(i)
        detected_cp, total_pulls = test_single_change_point_identification_discrete()
        print(f"Test {i+1}: Detected Change Point: {detected_cp}, Total Pulls: {total_pulls}")        