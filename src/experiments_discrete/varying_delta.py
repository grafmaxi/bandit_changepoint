from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time

from src.algorithms.localize_change_points import localize_change_points
from src.algorithms.mcp_track_and_stop import mcp_track_and_stop
from src.models.cp_bandit import CPBandit
from src.models.continuous_cp_bandit import Continuous_Bandit

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_discrete"

# ---------- Configuration ----------
num_mc = 100
num_workers = 4
num_arms = 2**8
s = 1/4
plateau_size = int(num_arms * s)
jump_size = 1
precision = 1 / (2 * num_arms)
delta_explore = .5  

log_inv_delta_values = [20, 30, 40, 50, 60, 70, 80, 90, 100]
delta_values = [np.exp(-x) for x in log_inv_delta_values]


# ---------- Worker ----------
def two_change_points(i):
    """Run all delta values for a single MC run (discrete + continuous)."""
    np.random.seed(i)
    left_pos = int(np.random.uniform(0, 0.5) * num_arms)
    gap = plateau_size
    sec_lens = [left_pos+1, gap, num_arms - left_pos - gap-1]
    means = [0, jump_size, 0]

    discrete_rows = []
    continuous_rows = []

    for delta in delta_values:
        # --- Discrete ---
        discrete_bandit = CPBandit(
            sec_lens, means,
            sampling_type='gaussian', parameter=1,
        )
        true_change_points = discrete_bandit.change_points
        estimated_change_points, budget = mcp_track_and_stop(
            discrete_bandit, delta, num_find=2,
        )
        success = int(
            min(true_change_points) == min(estimated_change_points)
            and max(true_change_points) == max(estimated_change_points)
        )
        discrete_rows.append({
            'run_id': i,
            'log_inv_delta': -np.log(delta),
            'gap': gap,
            'left_pos': left_pos,
            'true_cps': tuple(sorted(true_change_points)),
            'estimated_cps': tuple(sorted(estimated_change_points)),
            'budget': budget,
            'success': success,
        })

        # --- Continuous ---
        true_continuous_change_points = [
            (2 * true_change_points[0] + 1) / (2 * num_arms),
            (2 * true_change_points[1] + 1) / (2 * num_arms),
        ]
        continuous_bandit = Continuous_Bandit(
            change_points=true_continuous_change_points,
            values=means,
            sampling_type='gaussian', parameter=1,
        )
        (estimated_continuous_change_points, budget,
         detection_budget, estimation_budget,
         localization_budget, verification_budget) = localize_change_points(
            continuous_bandit,
            num_cps=2,
            delta=delta,
            delta_explore=delta_explore,
            precision=precision,
        )
        estimated_change_points = [
            int(estimated_continuous_change_points[0] * num_arms),
            int(estimated_continuous_change_points[1] * num_arms),
        ]
        success = int(
            min(true_change_points) == min(estimated_change_points)
            and max(true_change_points) == max(estimated_change_points)
        )
        continuous_rows.append({
            'run_id': i,
            'log_inv_delta': -np.log(delta),
            'gap': gap,
            'left_pos': left_pos,
            'true_cps': tuple(sorted(true_change_points)),
            'estimated_cps': tuple(sorted(estimated_change_points)),
            'budget': budget,
            'detection_budget': detection_budget,
            'estimation_budget': estimation_budget,
            'localization_budget': localization_budget,
            'verification_budget': verification_budget,
            'success': success,
        })

    return discrete_rows, continuous_rows


# ---------- Main ----------
if __name__ == "__main__":
    print(f"Tasks: {num_mc} MC runs x {len(delta_values)} log(1/delta) values = "
          f"{num_mc * len(delta_values)} single experiments per case.")
    print(f"Workers: {num_workers}")
 
    all_discrete = []
    all_continuous = []

    start_time = time.time()
    with Pool(num_workers) as pool:
        for d_rows, c_rows in tqdm(
            pool.imap_unordered(two_change_points, range(num_mc)),
            total=num_mc,
            desc="MC runs",
        ):
            all_discrete.extend(d_rows)
            all_continuous.extend(c_rows)
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    df_discrete = pd.DataFrame(all_discrete).sort_values(
        ['log_inv_delta', 'run_id']
    ).reset_index(drop=True)
    df_continuous = pd.DataFrame(all_continuous).sort_values(
        ['log_inv_delta', 'run_id']
    ).reset_index(drop=True)

    df_discrete.to_csv(
        RESULTS_DIR / "varying_delta_results_discrete.csv", index=False,
    )
    df_continuous.to_csv(
        RESULTS_DIR / "varying_delta_results_continuous.csv", index=False,
    )

    print(f"\nDiscrete:   {len(df_discrete)} rows "
          f"-> varying_delta_results_discrete.csv")
    print(f"Continuous: {len(df_continuous)} rows "
          f"-> varying_delta_results_continuous.csv")

    print("\n=== Average budget and success rate per log(1/delta) ===")
    print("\nDiscrete:")
    print(df_discrete.groupby('log_inv_delta').agg(
        mean_budget=('budget', 'mean'),
        success_rate=('success', 'mean'),
    ))
    print("\nContinuous:")
    print(df_continuous.groupby('log_inv_delta').agg(
        mean_budget=('budget', 'mean'),
        success_rate=('success', 'mean'),
    ))