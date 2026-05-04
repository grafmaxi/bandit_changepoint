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
num_mc = 1000
num_workers = 4
min_num_arms = 2**5
num_arms_iterations = 7
s = 1/4
jump_size = 1
delta = 0.05
delta_explore = 1


# ---------- Worker ----------
def two_change_points(i):
    """Run all num_arms values for a single MC run (discrete + continuous)."""
    np.random.seed(i)
    # Fix continuous ground truth once per MC run
    continuous_left_pos = np.random.uniform(0, 0.5)
    continuous_right_pos = continuous_left_pos + s
    true_continuous_change_points_fixed = [
        continuous_left_pos, continuous_right_pos,
    ]

    discrete_rows = []
    continuous_rows = []

    for iteration_index in range(num_arms_iterations):
        num_arms = min_num_arms * 2**iteration_index
        left_pos = int(continuous_left_pos * num_arms)
        gap = int(num_arms * s)
        sec_lens = [left_pos+1, gap, num_arms - left_pos - gap-1]
        means = [0, jump_size, 0]
        precision = 1 / (2 * num_arms)

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
            'num_arms': num_arms,
            'gap': gap,
            'left_pos': left_pos,
            'true_cps': tuple(sorted(true_change_points)),
            'estimated_cps': tuple(sorted(estimated_change_points)),
            'budget': budget,
            'success': success,
        })

        # --- Continuous ---
        continuous_bandit = Continuous_Bandit(
            change_points=true_continuous_change_points_fixed,
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
        # Discretize continuous estimates for comparison
        estimated_change_points_disc = sorted([
            int(x * num_arms) for x in estimated_continuous_change_points
        ])
        success = int(
            min(true_change_points) == min(estimated_change_points_disc)
            and max(true_change_points) == max(estimated_change_points_disc)
        )
        continuous_rows.append({
            'run_id': i,
            'num_arms': num_arms,
            'gap': gap,
            'left_pos': left_pos,
            'true_cps': tuple(sorted(true_change_points)),
            'estimated_cps': tuple(estimated_change_points_disc),
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
    print(f"Tasks: {num_mc} MC runs x {num_arms_iterations} num_arms values = "
          f"{num_mc * num_arms_iterations} single experiments per case.")
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
        ['num_arms', 'run_id']
    ).reset_index(drop=True)
    df_continuous = pd.DataFrame(all_continuous).sort_values(
        ['num_arms', 'run_id']
    ).reset_index(drop=True)

    df_discrete.to_csv(
        RESULTS_DIR / "varying_eta_results_discrete.csv", index=False,
    )
    df_continuous.to_csv(
        RESULTS_DIR / "varying_eta_results_continuous.csv", index=False,
    )

    print(f"\nDiscrete:   {len(df_discrete)} rows "
          f"-> varying_eta_results_discrete.csv")
    print(f"Continuous: {len(df_continuous)} rows "
          f"-> varying_eta_results_continuous.csv")

    print("\n=== Average budget and success rate per num_arms ===")
    print("\nDiscrete:")
    print(df_discrete.groupby('num_arms').agg(
        mean_budget=('budget', 'mean'),
        success_rate=('success', 'mean'),
    ))
    print("\nContinuous:")
    print(df_continuous.groupby('num_arms').agg(
        mean_budget=('budget', 'mean'),
        success_rate=('success', 'mean'),
    ))