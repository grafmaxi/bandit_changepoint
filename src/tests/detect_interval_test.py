from src.algorithms.detect_intervals import detect_intervals
from src.models.continuous_cp_bandit import Continuous_Bandit
import numpy as np # type: ignore


def test_detect_intervals():
    change_points = [0.1, 0.45, 0.55, 0.8, 0.9]
    values = [0, 1, 0, 1, 0, 1]
    bandit = Continuous_Bandit(change_points=change_points, values=values, sampling_type="gaussian", parameter=1.0)
    delta = 0.5
    budget = 10000
    intervals = detect_intervals(bandit, delta, budget)
    print(f"True change points: {change_points}")
    print(f"Detected intervals: {intervals}")

if __name__ == "__main__":
    for i in range(10):
        np.random.seed(i)
        print(f"Test run {i+1}")
        test_detect_intervals()
