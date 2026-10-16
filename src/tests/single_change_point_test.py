"""
Smoke tests for CSCPI (continuous) and SCPI (discrete) with one change point: prints results without checking them.
"""

from src.algorithms.continuous_single_change_point_identification import cscpi
from src.models.continuous_cp_bandit import Continuous_Bandit
from src.models.cp_bandit import CPBandit
from src.algorithms.scp_track_and_stop import scp_track_and_stop
import numpy as np

def test_single_change_point_identification_continuous():
    precision = 0.001
    confidence = 0.1
    cplist = [0.4]
    values = [0, 0.5]
    bandit = Continuous_Bandit(change_points=cplist, values=values, sampling_type="gaussian", parameter=1)

    detected_cp, tracked_budget = cscpi(bandit, confidence, precision)
    print(f"Continuous: Detected Change Point: {detected_cp}, Tracked Budget: {tracked_budget}")

def test_single_change_point_identification_discrete():
    delta = 0.1
    sec_len = [200, 300]
    sec_means = [0, .5]
    bandit = CPBandit(sec_len=sec_len, sec_means=sec_means, sampling_type="gaussian", parameter=1)

    detected_cp, total_pulls = scp_track_and_stop(bandit, delta)
    print(f"Discrete: Detected Change Point: {detected_cp}, Total Pulls: {total_pulls}")

if __name__ == "__main__":
    for i in range(10):
        np.random.seed(i)
        print(f"Test run {i+1}")
        test_single_change_point_identification_continuous()
        test_single_change_point_identification_discrete()
