import numpy as np
from src.models.cp_bandit import CPBandit

def test_binary_search_multiple_scenarios():
    # Define test scenarios
    scenarios = [
        {"means": [0, 1, 0], "lengths": [5, 1, 4], "name": "Positive jump, equal lengths"}
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
        estimated_ba = bandit.sequential_halving(budget)
        
        # Print results
        print(f"\nScenario: {scenario['name']}")
        print(f"Estimated best arm: {estimated_ba}")
if __name__ == "__main__":
    test_binary_search_multiple_scenarios()