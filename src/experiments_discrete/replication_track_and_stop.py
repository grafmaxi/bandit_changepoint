"""
Monte-Carlo experiment: replicating the three environments from
Lazzaro & Pike-Burke (2025), comparing MCPI (discrete Track-and-Stop)
with the continuous method across a range of delta values.

Environments (0-indexed change points):
  v1: mu = (2,2,2,2,2,2,1,1,1), N=1 CP at 5
  v2: mu = (2,2,2,2,2,2,4,4,4,4,4,4,4,0,0,0,0,0,0), N=2 CPs at 5, 12
  v3: mu = (2,2,3,3,3,3,1,1,4), N=3 CPs at 1, 5, 7

Parallelized, output: two CSVs for later plot creation.
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_discrete"

# ---------- Configuration ----------
num_mc = 1000
num_workers = 4
delta_explore = 1

# Range of log(1/delta) values to test (natural log)
log_inv_delta_values = [20, 30, 40, 50, 60, 70, 80, 90, 100]
delta_values = [np.exp(-x) for x in log_inv_delta_values]

# Environments from the paper (0-indexed change points)
environments = {
    'v1': {
        'means_by_section': [2, 1],
        'section_lengths': [6, 3],
        'num_cps': 1,
        'true_cps': [5],
    },
    'v2': {
        'means_by_section': [2, 4, 0],
        'section_lengths': [6, 7, 6],
        'num_cps': 2,
        'true_cps': [5, 12],
    },
    'v3': {
        'means_by_section': [2, 3, 1, 4],
        'section_lengths': [2, 4, 2, 1],
        'num_cps': 3,
        'true_cps': [1, 5, 7],
    },
}


# ---------- Worker ----------
def single_run(task):
    """Run one MC iteration for one (environment, delta) combination."""
    i, env_name, delta = task

    # Deterministic seed across Python sessions
    env_idx = list(environments.keys()).index(env_name)
    delta_idx = delta_values.index(delta)
    seed = i * 10000 + env_idx * 100 + delta_idx
    np.random.seed(seed)

    env = environments[env_name]
    means = env['means_by_section']
    lengths = env['section_lengths']
    num_cps = env['num_cps']
    true_cps = env['true_cps']
    num_arms = sum(lengths)
    precision = 1 / (2 * num_arms)

    # --- Discrete ---
    discrete_bandit = CPBandit(
        lengths, means,
        sampling_type='gaussian', parameter=1,
    )
    estimated_change_points, budget = mcp_track_and_stop(
        discrete_bandit, delta, num_find=num_cps,
    )
    success_d = int(sorted(estimated_change_points) == sorted(true_cps))
    discrete_row = {
        'run_id': i,
        'environment': env_name,
        'delta': delta,
        'log_inv_delta': -np.log(delta),
        'num_arms': num_arms,
        'num_cps': num_cps,
        'true_cps': tuple(sorted(true_cps)),
        'estimated_cps': tuple(sorted(estimated_change_points)),
        'budget': budget,
        'success': success_d,
    }

    # --- Continuous ---
    true_continuous_change_points = [
        (2 * cp + 1) / (2 * num_arms) for cp in true_cps
    ]
    continuous_bandit = Continuous_Bandit(
        change_points=true_continuous_change_points,
        values=means,
        sampling_type='gaussian', parameter=1,
    )
    (estimated_continuous_change_points, c_budget,
     detection_budget, estimation_budget,
     localization_budget, verification_budget) = localize_change_points(
        continuous_bandit,
        num_cps=num_cps,
        delta=delta,
        delta_explore=delta_explore,
        precision=precision,
    )
    estimated_change_points_disc = sorted([
        int(x * num_arms) for x in estimated_continuous_change_points
    ])
    success_c = int(estimated_change_points_disc == sorted(true_cps))
    continuous_row = {
        'run_id': i,
        'environment': env_name,
        'delta': delta,
        'log_inv_delta': -np.log(delta),
        'num_arms': num_arms,
        'num_cps': num_cps,
        'true_cps': tuple(sorted(true_cps)),
        'estimated_cps': tuple(estimated_change_points_disc),
        'budget': c_budget,
        'detection_budget': detection_budget,
        'estimation_budget': estimation_budget,
        'localization_budget': localization_budget,
        'verification_budget': verification_budget,
        'success': success_c,
    }

    return discrete_row, continuous_row


# ---------- Main ----------
if __name__ == "__main__":
    tasks = [
        (i, env_name, delta)
        for i in range(num_mc)
        for env_name in environments.keys()
        for delta in delta_values
    ]
    total = len(tasks)

    print(f"Tasks: {num_mc} MC runs x {len(environments)} environments "
          f"x {len(delta_values)} delta values = {total} per method.")
    print(f"Workers: {num_workers}")

    all_discrete = []
    all_continuous = []

    start_time = time.time()
    with Pool(num_workers) as pool:
        for d_row, c_row in tqdm(
            pool.imap_unordered(single_run, tasks),
            total=total,
            desc="Experiments",
        ):
            all_discrete.append(d_row)
            all_continuous.append(c_row)
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")

    df_discrete = pd.DataFrame(all_discrete).sort_values(
        ['environment', 'log_inv_delta', 'run_id']
    ).reset_index(drop=True)
    df_continuous = pd.DataFrame(all_continuous).sort_values(
        ['environment', 'log_inv_delta', 'run_id']
    ).reset_index(drop=True)

    df_discrete.to_csv(
        RESULTS_DIR / "paper_envs_results_discrete.csv", index=False,
    )
    df_continuous.to_csv(
        RESULTS_DIR / "paper_envs_results_continuous.csv", index=False,
    )

    print(f"\nDiscrete:   {len(df_discrete)} rows "
          f"-> paper_envs_results_discrete.csv")
    print(f"Continuous: {len(df_continuous)} rows "
          f"-> paper_envs_results_continuous.csv")

    print("\n=== Average budget and success rate per (environment, log(1/delta)) ===")
    print("\nDiscrete:")
    print(df_discrete.groupby(['environment', 'log_inv_delta']).agg(
        mean_budget=('budget', 'mean'),
        success_rate=('success', 'mean'),
    ))
    print("\nContinuous:")
    print(df_continuous.groupby(['environment', 'log_inv_delta']).agg(
        mean_budget=('budget', 'mean'),
        success_rate=('success', 'mean'),
    ))