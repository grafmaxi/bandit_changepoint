import numpy as np
from src.models.cp_bandit import CPBandit
from src.algorithms.change_point_detection import fixed_confidence

def test_change_point_detection():
    # Define test scenarios
    scenarios = [
        {"means": [0, -.03, .02, 0], "lengths": [40, 10, 20, 30], "name": "Three jumps"}
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
        change_points = fixed_confidence(
            bandit,
            min_cp_num=bandit.num_cp,
            confidence=0.05,
            track_budget=True
        )
        
        # Print results
        print(f"\nScenario: {scenario['name']}")
        print(f"Estimated change_points: {change_points}")
if __name__ == "__main__":
    test_change_point_detection()