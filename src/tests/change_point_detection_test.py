"""
Smoke tests for MCPI (discrete) and LCP (continuous) with five change points: prints results without checking them.
"""

import numpy as np
from src.models.cp_bandit import CPBandit
from src.models.continuous_cp_bandit import Continuous_Bandit
from src.algorithms.localize_change_points import localize_change_points
from src.algorithms.scp_track_and_stop import scp_track_and_stop
from src.algorithms.mcp_track_and_stop import mcp_track_and_stop

def test_change_point_detection_discrete():
    k = 2000
    scenarios = [
        {"means": [0, 1, 0, 1, 0, 1], "lengths": [k,k,k,k,k,k], "name": 'equidistant jumps'}
    ]
    
    
    for scenario in scenarios:
        bandit = CPBandit(
            sec_means=scenario["means"],
            sec_len=scenario["lengths"],
            sampling_type='gaussian',
            parameter=1 
        )
        
        [change_points, budget] = mcp_track_and_stop(bandit, delta=0.2, num_find=5)
        
        true_change_points = bandit.change_points
        print(f"True change_points: {true_change_points}")
        print(f"Estimated change_points: {change_points}")
        print(f"Budget used: {budget}")

def test_change_point_detection_continuous():
    scenarios = [
        {"change_points": [0.1, .3, .5, 0.7, 0.9], "values": [0, 1, 0, 1, 0, 1], "name": 'multiple jumps'}
    ]
    
    for scenario in scenarios:
        bandit = Continuous_Bandit(
            change_points=scenario["change_points"],
            values=scenario["values"],
            sampling_type='gaussian',
            parameter=1 
        )
        
        change_points, budget, *_ = localize_change_points(bandit, num_cps=5, delta=0.2, delta_explore=0.5, precision=0.0001)
        
        true_change_points = bandit.change_points
        print(f"True change_points: {true_change_points}")
        print(f"Estimated change_points: {change_points}")
        print(f"Budget used: {budget}")


if __name__ == "__main__":    
    for i in range(10):
        np.random.seed(i)
        print(i)
        test_change_point_detection_continuous()
        test_change_point_detection_discrete()