"""
Experiment 2 (Figure 2a): two change points with varying confidence level delta.
Compares LCP with MCPI. Output: two CSVs for later plot creation.
"""

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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_continuous"

# ---------- Configuration ----------
num_mc = 1000
jump_size = 1
precision = 2**(-8)
s = 1/4
delta_explore = 1
num_workers = 4

log_inv_delta_values = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
delta_values = [np.exp(-x) for x in log_inv_delta_values]


# ---------- Worker ----------
def varying_delta(i):
    """Run all delta values for a single MC run (continuous + discrete)."""
    np.random.seed(i)
    continuous_left_pos = np.random.uniform(0, 0.5)
    continuous_right_pos = continuous_left_pos + s
    true_continuous_cps = [continuous_left_pos, continuous_right_pos]
    means = [0, jump_size, 0]

    discrete_rows = []
    continuous_rows = []

    for delta in delta_values:
        # --- Continuous ---
        continuous_bandit = Continuous_Bandit(
            change_points=true_continuous_cps,
            values=means,
            sampling_type='gaussian', parameter=1,
        )
        start_time_continuous = time.time()
        (estimated_continuous_cps, c_budget,
         detection_budget, estimation_budget,
         localization_budget, verification_budget) = localize_change_points(
            continuous_bandit,
            num_cps=2,
            delta=delta,
            delta_explore=delta_explore,
            precision=precision,
        )
        runtime_continuous = time.time() - start_time_continuous
        continuous_success = int(
            abs(true_continuous_cps[0] - estimated_continuous_cps[0]) <= precision
            and abs(true_continuous_cps[1] - estimated_continuous_cps[1]) <= precision
        )
        continuous_deviation = (
            abs(true_continuous_cps[0] - estimated_continuous_cps[0])
            + abs(true_continuous_cps[1] - estimated_continuous_cps[1])
        )
        continuous_rows.append({
            'run_id': i,
            'log_inv_delta': -np.log(delta),
            's': s,
            'left_pos': continuous_left_pos,
            'true_cps': tuple(true_continuous_cps),
            'estimated_cps': tuple(estimated_continuous_cps),
            'budget': c_budget,
            'detection_budget': detection_budget,
            'estimation_budget': estimation_budget,
            'localization_budget': localization_budget,
            'verification_budget': verification_budget,
            'success': continuous_success,
            'deviation': continuous_deviation,
            'runtime': runtime_continuous,
        })

        # --- Discrete ---
        num_arms = int(1/precision) + 1
        discrete_left_pos = int(continuous_left_pos/precision) + 1
        discrete_right_pos = int(continuous_right_pos/precision) + 1
        sec_lens = [
            discrete_left_pos,
            discrete_right_pos - discrete_left_pos,
            num_arms - discrete_right_pos,
        ]

        discrete_bandit = CPBandit(
            sec_lens, means,
            sampling_type='gaussian', parameter=1,
        )
        start_time_discrete = time.time()
        estimated_change_points, d_budget = mcp_track_and_stop(
            discrete_bandit, delta, num_find=2,
        )
        runtime_discrete = time.time() - start_time_discrete
        # Map discrete estimates back to continuous coordinates
        discrete_to_continuous_estimated_cps = sorted([
            x * precision for x in estimated_change_points
        ])
        discrete_success = int(
            abs(true_continuous_cps[0] - discrete_to_continuous_estimated_cps[0]) <= precision
            and abs(true_continuous_cps[1] - discrete_to_continuous_estimated_cps[1]) <= precision
        )
        deviation = (
            abs(true_continuous_cps[0] - discrete_to_continuous_estimated_cps[0])
            + abs(true_continuous_cps[1] - discrete_to_continuous_estimated_cps[1])
        )
        discrete_rows.append({
            'run_id': i,
            'log_inv_delta': -np.log(delta),
            's': s,
            'left_pos': discrete_left_pos,
            'true_cps': tuple(true_continuous_cps),
            'estimated_cps': tuple(discrete_to_continuous_estimated_cps),
            'budget': d_budget,
            'success': discrete_success,
            'deviation': deviation,
            'runtime': runtime_discrete,
        })

    return continuous_rows, discrete_rows


# ---------- Main ----------
if __name__ == "__main__":
    print(f"Tasks: {num_mc} MC runs x {len(delta_values)} log(1/delta) values = "
          f"{num_mc * len(delta_values)} single experiments per case.")
    print(f"Workers: {num_workers}")

    all_discrete = []
    all_continuous = []

    start_time = time.time()
    with Pool(num_workers) as pool:
        for c_rows, d_rows in tqdm(
            pool.imap_unordered(varying_delta, range(num_mc)),
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
    total_lcp = df_continuous['runtime'].sum() if 'runtime' in df_continuous.columns else 0
    total_mcpi = df_discrete['runtime'].sum() if 'runtime' in df_discrete.columns else 0
    print(f"Total runtime LCP (sum across all runs): {total_lcp:.2f} seconds")
    print(f"Total runtime MCPI (sum across all runs): {total_mcpi:.2f} seconds")    

    print(f"\nDiscrete:   {len(df_discrete)} rows "
          f"-> varying_delta_results_discrete.csv")
    print(f"Continuous: {len(df_continuous)} rows "
          f"-> varying_delta_results_continuous.csv")

    print("\n=== Average budget, success rate, and deviation per log(1/delta) ===")
    print("\nDiscrete:")
    print(df_discrete.groupby('log_inv_delta').agg(
        mean_budget=('budget', 'mean'),
        success_rate=('success', 'mean'),
        mean_deviation=('deviation', 'mean'),
    ))
    print("\nContinuous:")
    print(df_continuous.groupby('log_inv_delta').agg(
        mean_budget=('budget', 'mean'),
        success_rate=('success', 'mean'),
        mean_deviation=('deviation', 'mean'),
    ))