from typing import List, Optional
from .bandit import Bandit

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