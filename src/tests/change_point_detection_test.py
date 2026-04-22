import numpy as np
from src.models.cp_bandit import CPBandit
from src.algorithms.scp_track_and_stop import scp_track_and_stop
from src.algorithms.mcp_track_and_stop import mcp_track_and_stop

def test_change_point_detection():
    # Define test scenarios
    scenarios = [
        {"means": [0, 1, 0], "lengths": [900, 2, 900], "name": 'single jump'}
    ]
    
    
    for scenario in scenarios:
        # Create CPBandit
        bandit = CPBandit(
            sec_means=scenario["means"],
            sec_len=scenario["lengths"],
            sampling_type='gaussian',
            parameter=1 
        )
        
        # Apply binary search
        [change_points, budget] = mcp_track_and_stop(bandit, delta=0.1, num_find=2)  # Get estimated change point
        
        # Print results
        # print(f"\nScenario: {scenario['single jump']}")
        true_change_points = bandit.change_points  # Get true change point from bandit
        print(f"True change_points: {true_change_points}")
        print(f"Estimated change_points: {change_points}")
        print(f"Budget used: {budget}")
if __name__ == "__main__":    
    for i in range(10):
        np.random.seed(i)
        print(i)
        test_change_point_detection()