import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt # type: ignore

results_dir = Path(__file__).resolve().parents[1] / "results"

# Backtracking CPs
with open(results_dir / "scpd_backtracking_cps.csv", 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    backtracking_cps = np.array([list(map(float, row)) for row in reader])

# Backtracking Budgets
with open(results_dir / "scpd_backtracking_budgets.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Header überspringen
    backtracking_budgets = np.array([list(map(float, row)) for row in reader])

# CSPI CPs
with open(results_dir / "scpd_cspi_cps.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    cspi_cps = np.array([list(map(float, row)) for row in reader])

# CSPI Budgets
with open(results_dir / "scpd_cspi_budgets.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    cspi_budgets = np.array([list(map(float, row)) for row in reader])

mean_backtracking_budgets = np.mean(backtracking_budgets, axis=0)
mean_cspi_budgets = np.mean(cspi_budgets, axis=0)
print("Backtracking Budgets:", mean_backtracking_budgets)
print("SCPI Budgets:", mean_cspi_budgets)
    
# Plotting
num_arms_list = [int(h.split('_')[0]) for h in header]
plt.figure(figsize=(10, 5))
plt.plot(num_arms_list, mean_backtracking_budgets, label="Backtracking", marker='o')
plt.plot(num_arms_list, mean_cspi_budgets, label="SCPI", marker='s')
plt.xlabel("Number of Arms")
plt.ylabel("Mean Budget")
plt.legend()
plt.title("Mean Budget vs Number of Arms") 
plt.grid(True)
plt.savefig(results_dir / "scpd_budgets_plot.png")
plt.show()