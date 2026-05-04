"""
Plot mean budgets against log(1/delta) for the varying-delta experiment,
comparing discrete (MCPI) and continuous methods.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_discrete"

# ---------- Load data ----------
df_discrete = pd.read_csv(RESULTS_DIR / "varying_delta_results_discrete.csv")
df_continuous = pd.read_csv(RESULTS_DIR / "varying_delta_results_continuous.csv")

# ---------- Aggregate ----------
agg_d = df_discrete.groupby('log_inv_delta')['budget'].agg(
    mean='mean',
    q05=lambda x: np.quantile(x, 0.05),
    q95=lambda x: np.quantile(x, 0.95),
)
agg_c = df_continuous.groupby('log_inv_delta')['budget'].agg(
    mean='mean',
    q05=lambda x: np.quantile(x, 0.05),
    q95=lambda x: np.quantile(x, 0.95),
)

# ---------- Plot: budget vs. log(1/delta) ----------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(agg_d.index, agg_d['mean'],
        marker='o', label='Discrete (MCPI)', color='tab:blue')
ax.fill_between(agg_d.index, agg_d['q05'], agg_d['q95'],
                alpha=0.2, color='tab:blue')

ax.plot(agg_c.index, agg_c['mean'],
        marker='s', label='Continuous', color='tab:red')
ax.fill_between(agg_c.index, agg_c['q05'], agg_c['q95'],
                alpha=0.2, color='tab:red')

ax.set_xlabel(r'$\log(1/\delta)$')
ax.set_ylabel('Mean budget (stopping time)')
ax.set_title('Mean sample complexity vs. confidence level')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'varying_delta_budget_vs_delta.pdf')
plt.savefig(RESULTS_DIR / 'varying_delta_budget_vs_delta.png', dpi=150)
plt.show()


# ---------- Optional: success rate plot ----------
fig, ax = plt.subplots(figsize=(7, 5))

succ_d = df_discrete.groupby('log_inv_delta')['success'].mean()
succ_c = df_continuous.groupby('log_inv_delta')['success'].mean()

ax.plot(succ_d.index, succ_d.values,
        marker='o', label='Discrete (MCPI)', color='tab:blue')
ax.plot(succ_c.index, succ_c.values,
        marker='s', label='Continuous', color='tab:red')

ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel(r'$\log(1/\delta)$')
ax.set_ylabel('Success rate')
ax.set_ylim(-0.05, 1.05)
ax.set_title('Success rate vs. confidence level')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'varying_delta_success_vs_delta.pdf')
plt.savefig(RESULTS_DIR / 'varying_delta_success_vs_delta.png', dpi=150)
plt.show()