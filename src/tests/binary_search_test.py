import numpy as np
from src.models.cp_bandit import CPBandit

def test_binary_search_multiple_scenarios():
    # Define test scenarios
    scenarios = [
        {"means": [0, 1], "lengths": [5, 5], "name": "Positive jump, equal lengths"},
        {"means": [0, -1], "lengths": [5, 5], "name": "Negative jump, equal lengths"},
        {"means": [0, 1], "lengths": [4, 6], "name": "Positive jump, unequal lengths"},
        {"means": [0, -1], "lengths": [4, 6], "name": "Negative jump, unequal lengths"}
    ]
    
    budget = 100
    
    for scenario in scenarios:
        # Create CPBandit
        bandit = CPBandit(
            sec_means=scenario["means"],
            sec_len=scenario["lengths"],
            sampling_type='gaussian',
            parameter=0.1
        )
        
        # Apply binary search
        estimated_cp = bandit.bin_search(budget, left = 2, right = 8)
        true_cp = bandit.change_points[0]  # First changepoint
        
        # Print results
        print(f"\nScenario: {scenario['name']}")
        print(f"True changepoint: {true_cp}")
        print(f"Estimated changepoint: {estimated_cp}")
        print(f"Absolute error: {abs(true_cp - estimated_cp)}")

if __name__ == "__main__":
    test_binary_search_multiple_scenarios()