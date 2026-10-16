"""
Experiment 5 (Figure 4): ten change points with varying delta and eta.
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
num_mc = 100
num_cp = 10
s = 0.25 / num_cp
base_precision = 0.1 * s
num_precision_iterations = 3
jump_size = 1
delta_explore = 1
num_workers = 4

log_inv_delta_values = [20, 40, 60, 80, 100]
delta_values = [np.exp(-x) for x in log_inv_delta_values]


# ---------- Worker ----------
def many_change_points(i):
    """Run all (precision, delta) combinations for a single MC run."""
    np.random.seed(i)
    # Random left starting position; ensure all num_cp CPs fit inside [0, 1)
    continuous_left_pos = np.random.uniform(0, 0.5)
    # CPs at regular spacing s starting from continuous_left_pos
    true_continuous_cps = [
        continuous_left_pos + (k + 1) * s for k in range(num_cp)
    ]
    # Alternating means between sections: 0, jump_size, 0, jump_size, ...
    means = [
        0 if k % 2 == 0 else jump_size for k in range(num_cp + 1)
    ]

    discrete_rows = []
    continuous_rows = []

    for power in range(num_precision_iterations):
        precision = base_precision / 2 ** (power + 1)

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
                num_cps=num_cp,
                delta=delta,
                delta_explore=delta_explore,
                precision=precision,
            )
            runtime_continuous = time.time() - start_time_continuous
            sorted_continuous_estimates = sorted(estimated_continuous_cps)
            continuous_success = int(all(
                abs(true_continuous_cps[k] - sorted_continuous_estimates[k]) <= precision
                for k in range(num_cp)
            ))
            continuous_deviation = sum(
                abs(true_continuous_cps[k] - sorted_continuous_estimates[k])
                for k in range(num_cp)
            )
            continuous_rows.append({
                'run_id': i,
                'precision': precision,
                'log_inv_delta': -np.log(delta),
                'left_pos': continuous_left_pos,
                'true_cps': tuple(true_continuous_cps),
                'estimated_cps': tuple(sorted_continuous_estimates),
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
            discrete_cps = [
                int(cp / precision) + 1 for cp in true_continuous_cps
            ]
            sec_lens = [discrete_cps[0]]
            for k in range(num_cp - 1):
                sec_lens.append(discrete_cps[k+1] - discrete_cps[k])
            sec_lens.append(num_arms - discrete_cps[-1])

            discrete_bandit = CPBandit(
                sec_lens, means,
                sampling_type='gaussian', parameter=1,
            )
            start_time_discrete = time.time()
            estimated_change_points, d_budget = mcp_track_and_stop(
                discrete_bandit, delta, num_find=num_cp,
            )
            runtime_discrete = time.time() - start_time_discrete
            discrete_to_continuous_estimated_cps = sorted([
                x * precision for x in estimated_change_points
            ])
            discrete_success = int(all(
                abs(true_continuous_cps[k] - discrete_to_continuous_estimated_cps[k]) <= precision
                for k in range(num_cp)
            ))
            discrete_deviation = sum(
                abs(true_continuous_cps[k] - discrete_to_continuous_estimated_cps[k])
                for k in range(num_cp)
            )
            discrete_rows.append({
                'run_id': i,
                'precision': precision,
                'log_inv_delta': -np.log(delta),
                'num_arms': num_arms,
                'left_pos': continuous_left_pos,
                'true_cps': tuple(true_continuous_cps),
                'estimated_cps': tuple(discrete_to_continuous_estimated_cps),
                'budget': d_budget,
                'success': discrete_success,
                'deviation': discrete_deviation,
                'runtime': runtime_discrete,
            })

    return continuous_rows, discrete_rows


# ---------- Main ----------
if __name__ == "__main__":
    total_per_method = num_mc * num_precision_iterations * len(delta_values)
    print(f"Tasks: {num_mc} MC runs x {num_precision_iterations} precision values "
          f"x {len(delta_values)} log(1/delta) values = "
          f"{total_per_method} single experiments per case.")
    print(f"Workers: {num_workers}")

    all_discrete = []
    all_continuous = []

    start_time = time.time()
    with Pool(num_workers) as pool:
        for c_rows, d_rows in tqdm(
            pool.imap_unordered(many_change_points, range(num_mc)),
            total=num_mc,
            desc="MC runs",
        ):
            all_continuous.extend(c_rows)
            all_discrete.extend(d_rows)
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")


    df_discrete = pd.DataFrame(all_discrete).sort_values(
        ['precision', 'log_inv_delta', 'run_id']
    ).reset_index(drop=True)
    df_continuous = pd.DataFrame(all_continuous).sort_values(
        ['precision', 'log_inv_delta', 'run_id']
    ).reset_index(drop=True)

    

    df_discrete.to_csv(RESULTS_DIR / "many_cps_results_discrete.csv", index=False)
    df_continuous.to_csv(RESULTS_DIR / "many_cps_results_continuous.csv", index=False)
    total_lcp = df_continuous['runtime'].sum() if 'runtime' in df_continuous.columns else 0
    total_mcpi = df_discrete['runtime'].sum() if 'runtime' in df_discrete.columns else 0
    print(f"Total runtime LCP (sum across all runs): {total_lcp:.2f} seconds")
    print(f"Total runtime MCPI (sum across all runs): {total_mcpi:.2f} seconds")
    print(f"\nDiscrete:   {len(df_discrete)} rows -> many_cps_results_discrete.csv")
    print(f"Continuous: {len(df_continuous)} rows -> many_cps_results_continuous.csv")

    # ---------- Matrix-style summaries ----------
    def print_matrices(df, label):
        print(f"\n=== {label} ===")
        agg = df.groupby(['precision', 'log_inv_delta']).agg(
            mean_budget=('budget', 'mean'),
            success_rate=('success', 'mean'),
            mean_deviation=('deviation', 'mean'),
        )
        for col in ['mean_budget', 'success_rate', 'mean_deviation']:
            print(f"\n{col} (rows: precision, cols: log(1/delta)):")
            matrix = agg[col].unstack(level='log_inv_delta')
            print(matrix.to_string(float_format=lambda x: f"{x:.4g}"))

    print_matrices(df_discrete, "Discrete")
    print_matrices(df_continuous, "Continuous")