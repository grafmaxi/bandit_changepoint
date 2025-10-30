import numpy as np
from src.models.cp_bandit import CPBandit

def test_binary_search_multiple_scenarios():
    # Define test scenarios
    scenarios = [
        {"means": [0, -3, 2, 0], "lengths": [4, 1, 2, 3], "detected": [4,7],  "name": "One jump, short second plateau"}
    ]
    
    budget = 1000
    
    for scenario in scenarios:
        # Create CPBandit
        bandit = CPBandit(
            sec_means=scenario["means"],
            sec_len=scenario["lengths"],
            sampling_type='gaussian',
            parameter=1 
        )
        
        # Apply binary search
        estimated_ba = bandit.sequential_halving(budget, cplist=scenario["detected"])
        
        # Print results
        print(f"\nScenario: {scenario['name']}")
        print(f"Estimated best arm: {estimated_ba}")
if __name__ == "__main__":
    test_binary_search_multiple_scenarios()