"""
Finite-armed bandit with Gaussian or Bernoulli rewards.
"""

from typing import List, Optional, Union
import numpy as np
from numpy.typing import NDArray

class Bandit:
    def __init__(
            self,
            arms: List[float],
            sampling_type: Optional[str] = "gaussian",
            parameter: Optional[float] = 1.0
    ) -> None:
        """Initialize a basic bandit.

        Args:
            arms: List of arm means.
            sampling_type: distribution from which to sample (gaussian or bernoulli).
            parameter: standard deviation for gaussian, ignored for bernoulli.
        """
        self.arms = arms
        self.sampling_type = sampling_type
        self.parameter = parameter
        self.num_arms = len(arms)
        
        self.valid_sampling_types = {"gaussian", "bernoulli"}
        if self.sampling_type not in self.valid_sampling_types:
            raise ValueError(f"Unsupported sampling type: {self.sampling_type}")
        if sampling_type == "gaussian" and (parameter is None or parameter <= 0):
            raise ValueError("Parameter must be a positive float for gaussian sampling.")
        if sampling_type == "bernoulli" and (any(arm < 0 or arm > 1 for arm in arms)):
            raise ValueError("All arm means must be in [0, 1] for bernoulli sampling.")

    def pull(self, i: int, ntimes: Optional[int] = 1) -> Union[float, NDArray[np.float64]]:
        """Sample rewards from arm i according to the specified distribution.
        
        Args:
            i: Index of the arm to pull (0 <= i < num_arms)
            ntimes: Number of independent samples to draw (default: 1)
            
        Returns:
            float: Single sampled reward if ntimes=1
            NDArray: Array of sampled rewards if ntimes>1
            
        Raises:
            ValueError: If i is out of bounds or ntimes < 1
        """
        if not 0 <= i < self.num_arms:
            raise ValueError(f"Arm index must be between 0 and {self.num_arms-1}")
        if ntimes < 1:
            raise ValueError("ntimes must be at least 1")
        
        mean = self.arms[i]
        
        if self.sampling_type == 'gaussian':
            samples = np.random.normal(mean, self.parameter, size=ntimes)
        elif self.sampling_type == 'bernoulli':
            samples = np.random.binomial(1, mean, size=ntimes).astype(float)
            
        return float(samples[0]) if ntimes == 1 else samples