from src.models.bandit import Bandit
from src.models.cp_bandit import CPBandit
from src.models.continuous_cp_bandit import Continuous_Bandit

# test instance of a bandt
arms1 = [0.1, 0.5, 0.9]
bandit1 = Bandit(arms=arms1, sampling_type="gaussian", parameter=1.0)
print("Bandit instance created with arms:", bandit1.arms)


# test instance of a cp bandit
sec_len2 = [3,3,3]
sec_means2 = [0,1,0]
cp_bandit2 = CPBandit(sec_len=sec_len2, sec_means=sec_means2)
print("CP Bandit instance created with sections length:", cp_bandit2.sec_len)
num_arms2 = cp_bandit2.num_arms
print("CP Bandit instance created with number of arms:", num_arms2)
arms2 = cp_bandit2.arms
print("CP Bandit instance created with arms:", arms2)
change_points2 = cp_bandit2.change_points
print("CP Bandit instance created with change points:", change_points2)

# test instance of a continuous cp bandit
change_points3 = [0.3, 0.6]
values3 = [0, 1, 0]
cont_cp_bandit3 = Continuous_Bandit(change_points=change_points3, values=values3, sampling_type="gaussian", parameter=1.0)
print("Continuous CP Bandit instance created with change points:", cont_cp_bandit3.change_points)
