from src.models.cp_bandit import CPBandit
from src.algorithms.scp_track_and_stop import scp_track_and_stop
from src.algorithms.continuous_single_change_point_identification import cscpi
import numpy as np # type: ignore
from joblib import Parallel, delayed
import csv
from pathlib import Path

def run_single_iteration(iteration, num_arm_list, cp_pos, delta, sampling_type, h):
    iter_results = iteration_track_and_stop(
        iteration, num_arm_list, cp_pos, delta, sampling_type, h
    )
    iter_results_cspi = iteration_cspi(
        iteration, num_arm_list, cp_pos, delta, sampling_type, h
    )
    return iter_results[0], iter_results[1], iter_results_cspi[0], iter_results_cspi[1]

def run_single_change_point_discrete_experiment():
    monte_carlo_runs = 1
    num_arm_list = list(range(5, 151, 5))
    cp_pos = .5
    delta = 0.1
    sampling_type = "gaussian"
    h = 0.5
    
    n_jobs = -1
    
    # Parallelisierte Ausführung
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_iteration)(
            iteration, num_arm_list, cp_pos, delta, sampling_type, h
        )
        for iteration in range(monte_carlo_runs)
    )
    
    # Entpacken der Ergebnisse
    backtracking_cps = [r[0] for r in results]
    backtracking_budgets = [r[1] for r in results]
    cspi_cps = [r[2] for r in results]
    cspi_budgets = [r[3] for r in results]
    
    return backtracking_cps, backtracking_budgets, cspi_cps, cspi_budgets, num_arm_list

def iteration_track_and_stop(iteration: int,
                             num_arms: list[int],
                             cp_pos: float,
                             delta: float,
                             sampling_type: str,
                             h: float
                             ):
    np.random.seed(iteration)
    iter_results = [[],[]]
    for num_arm in num_arms:
        sec_len = [int(num_arm*cp_pos), int(num_arm*(1-cp_pos))]
        sec_means = [0,h]
        sampling_type = sampling_type
        parameter = 1.0
        iter_bandit = CPBandit(sec_len, sec_means, sampling_type, parameter)
        [est_cp, est_budget] = scp_track_and_stop(iter_bandit, delta)
        iter_results[0].append(est_cp)
        iter_results[1].append(est_budget)
    return iter_results

def iteration_cspi(iteration: int,
                   num_arms: list[int],
                   cp_pos: float,
                   delta: float,
                   sampling_type: str,
                   h: float
                   ):
    np.random.seed(iteration)
    iter_results = [[],[]]
    for num_arm in num_arms:
        sec_len = [int(num_arm*cp_pos), int(num_arm*(1-cp_pos))]
        sec_means = [0,h]
        sampling_type = sampling_type
        parameter = 1.0
        iter_discrete_bandit = CPBandit(sec_len, sec_means, sampling_type, parameter)
        iter_continuous_bandit = iter_discrete_bandit.continuous_transform()
        [est_cp, est_budget] = cscpi(iter_continuous_bandit, delta, 1/(2*num_arm))
        iter_results[0].append(int(num_arm*est_cp-1))
        iter_results[1].append(est_budget)
    return iter_results

if __name__ == "__main__":
    # Experiment ausführen
    backtracking_cps, backtracking_budgets, cspi_cps, cspi_budgets, num_arm_list = run_single_change_point_discrete_experiment()
    
    # Pfad zum results Ordner
    results_dir = Path(__file__).resolve().parents[1] / "results_discrete"
    results_dir.mkdir(exist_ok=True)  # Ordner erstellen falls nicht vorhanden
    
    # Header: Spaltennamen sind die Anzahl der Arms
    header = [f"{num_arms}_arms" for num_arms in num_arm_list]
    
    # CSV 1: Backtracking Change Points
    with open(results_dir / "scpd_backtracking_cps.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Header schreiben
        for iteration_result in backtracking_cps:
            writer.writerow(iteration_result)
    
    # CSV 2: Backtracking Budgets
    with open(results_dir / "scpd_backtracking_budgets.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for iteration_result in backtracking_budgets:
            writer.writerow(iteration_result)
    
    # CSV 3: CSPI Change Points
    with open(results_dir / "scpd_cspi_cps.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for iteration_result in cspi_cps:
            writer.writerow(iteration_result)
    
    # CSV 4: CSPI Budgets
    with open(results_dir / "scpd_cspi_budgets.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for iteration_result in cspi_budgets:
            writer.writerow(iteration_result)
    
    print(f"Experiment abgeschlossen. 4 CSV-Dateien gespeichert in: {results_dir}")
    print(f"- {len(backtracking_cps)} Monte Carlo Iterationen")
    print(f"- {len(num_arm_list)} verschiedene Armzahlen ({num_arm_list[0]} bis {num_arm_list[-1]})")