from typing import List, Optional
from .bandit import Bandit
import numpy as np # type: ignore

class CPBandit(Bandit):
    def __init__(
            self,
            sec_len: List[int],
            sec_means: List[float],
            sampling_type: str = 'gaussian',
            parameter: Optional[float] = 1.0
    ) -> None:
        self.sec_len = sec_len
        self.sec_means = sec_means
        self.num_cp = len(sec_len) - 1
        if len(sec_means)-1 != self.num_cp:
            raise ValueError("Length of sec_means must be equal to length of sec_len")
        arms = []
        for i in range(self.num_cp + 1):
            arms.extend([sec_means[i]] * sec_len[i])
        
        self.change_points = []
        self.differences = []
        for i in range(self.num_cp):
            cp_index = sum(sec_len[:i+1])
            self.change_points.append(cp_index)
            self.differences.append(sec_means[i+1] - sec_means[i])

        super().__init__(arms = arms , sampling_type = sampling_type, parameter = parameter)

    def bin_search(
            self,
            budget: int,
            left: Optional[int] = None,
            right: Optional[int] = None
    ) -> int:
        left = 0 if left is None else left
        right = self.num_arms - 1 if right is None else right

        section_length = right + 1 - left
        step_budget = int(budget / (3*np.log2(section_length)))
        if step_budget < 1:
            raise ValueError("Budget too small for binary search")
        if section_length <= 2:
            raise ValueError("Section length too small for binary search")
        mid = left + int((section_length-1) / 2)
        while right - left > 1:
            left_mean = np.mean(self.pull(left, step_budget))
            mid_mean = np.mean(self.pull(mid, step_budget))
            right_mean = np.mean(self.pull(right, step_budget))
            if abs(left_mean - mid_mean) > abs(right_mean - mid_mean):
                right = mid
            else:
                left = mid
            mid = left + int((right - left) / 2)
        return right
    
    def sequential_halving(
            self,
            budget: int,
            left: Optional[int] = None,
            right: Optional[int] = None,
            grid_dist: Optional[int] = 1
            ) -> int:
        left = 0 if left is None else left
        right = self.num_arms - 1 if right is None else right

        section_length = right - left + 1
        if grid_dist < 1 or grid_dist >= section_length:
            raise ValueError("grid_dist must be at least 1 and smaller than the section length")
        

        grid_points = [left + grid_dist]
        while grid_points[-1] + grid_dist <= right:
            grid_points.append(grid_points[-1] + grid_dist)
        
        grid_size = len(grid_points)
        depth = int(np.ceil(np.log2(grid_size)))
        
        if budget / ( 2* depth * grid_size) < 1:
            raise ValueError("Budget too small for sequential halving with given grid_dist")
        
        for i in range(depth):
            num_sample = int(2**i * budget / ( 2* depth * grid_size) )
            diffs = []
            for gp in grid_points:
                diffs.append(- abs( np.mean(self.pull(gp, num_sample)) - np.mean(self.pull(left, num_sample))))
            index_sort = np.argsort(diffs)
            grid_points = [grid_points[index_sort[j]] for j in range(2**(depth -i -1))]
        
        return grid_points[0] 
    
    def test_differences(
            self,
            budget: int,
            threshold: float,
            i: int,
            j: Optional[int] = None
                         ) -> bool:
        if j is None:
            if i == 0:
                raise ValueError("j must be provided when i is 0")
            j = i - 1
        if budget < 2:
            raise ValueError("Budget must be at least 2")
        imean = np.mean(self.pull(i, budget // 2))
        jmean = np.mean(self.pull(j, budget // 2))
        return abs(imean - jmean) > threshold