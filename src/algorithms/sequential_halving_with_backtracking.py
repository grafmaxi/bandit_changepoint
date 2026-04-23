import numpy as np # type: ignore
from src.models.continuous_cp_bandit import Continuous_Bandit

class tree_node:
    def __init__(self, left_boundary, l, m, r, right_boundary):
        self.left_boundary = left_boundary
        self.l = l
        self.m = m
        self.r= r
        self.right_boundary = right_boundary

    def left(self):
        return tree_node(self.left_boundary, self.l, (self.l + self.m) / 2, self.m, self.right_boundary)
    
    def right(self):
        return tree_node(self.left_boundary, self.m, (self.m + self.r) / 2, self.r, self.right_boundary)
    
def shb(
        bandit,
        budget: int,
        precision: float,
        interval = (0, 1),
        track_budget = False
) :
    left, right = interval
    if right - left <= 2*precision:
        if track_budget:
            return (left + right) / 2, 0
        else:
            return (left + right) / 2
    root = tree_node(left, left, (left + right) / 2, right, right)
    num_steps = int(np.ceil(6*np.log(1/precision)))
    step_budget = int(budget/(5*num_steps))
    tracked_budget = 0
    if step_budget == 0:
        return (left + right) / 2, 0
    history = [root]
    for i in range(num_steps):
        if step_budget == 1:
            mean_lb = bandit.pull(root.left_boundary, step_budget)
            mean_l = bandit.pull(root.l, step_budget)
            mean_m = bandit.pull(root.m, step_budget)
            mean_r = bandit.pull(root.r, step_budget)
            mean_rb = bandit.pull(root.right_boundary, step_budget)
        else:
            mean_lb = bandit.pull(root.left_boundary, step_budget).mean()
            mean_l = bandit.pull(root.l, step_budget).mean()
            mean_m = bandit.pull(root.m, step_budget).mean()
            mean_r = bandit.pull(root.r, step_budget).mean()
            mean_rb = bandit.pull(root.right_boundary, step_budget).mean()
        tracked_budget += 5*step_budget
        if abs((mean_lb+mean_l)/2-(mean_r+mean_rb)/2) < (3/4) * max(abs((mean_lb+mean_l+mean_r)/3 - mean_rb), abs(mean_lb - (mean_l+mean_r+mean_rb)/3)):
            if len (history) > 1:
                history.pop()
                root = history[-1]
        elif abs(mean_l-mean_m) > abs(mean_r - mean_m):
            root = root.left()
            history.append(root)
        else:
            root = root.right()
            history.append(root)
    if track_budget:
        return root.m, tracked_budget
    else:
        return root.m   