"""
Plot mean budgets against log(1/delta) for the three paper environments,
comparing discrete (MCPI) and continuous methods.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_discrete"

# ---------- Load data ----------
df_discrete = pd.read_csv(RESULTS_DIR / "paper_envs_results_discrete.csv")
df_continuous = pd.read_csv(RESULTS_DIR / "paper_envs_results_continuous.csv")

environments = sorted(df_discrete['environment'].unique())

# ---------- Plot: one subplot per environment ----------
fig, axes = plt.subplots(1, len(environments),
                         figsize=(5 * len(environments), 4.5),
                         sharex=True)
if len(environments) == 1:
    axes = [axes]

for ax, env in zip(axes, environments):
    d_env = df_discrete[df_discrete['environment'] == env]
    c_env = df_continuous[df_continuous['environment'] == env]

    agg_d = d_env.groupby('log_inv_delta')['budget'].agg(
        mean='mean',
        q05=lambda x: np.quantile(x, 0.05),
        q95=lambda x: np.quantile(x, 0.95),
    )
    agg_c = c_env.groupby('log_inv_delta')['budget'].agg(
        mean='mean',
        q05=lambda x: np.quantile(x, 0.05),
        q95=lambda x: np.quantile(x, 0.95),
    )

    ax.plot(agg_d.index, agg_d['mean'],
            marker='o', label='Discrete (MCPI)', color='tab:blue')
    ax.fill_between(agg_d.index, agg_d['q05'], agg_d['q95'],
                    alpha=0.2, color='tab:blue')

#    ax.plot(agg_c.index, agg_c['mean'],
#            marker='s', label='Continuous', color='tab:red')
#    ax.fill_between(agg_c.index, agg_c['q05'], agg_c['q95'],
#                    alpha=0.2, color='tab:red')

    num_arms = d_env['num_arms'].iloc[0]
    num_cps = d_env['num_cps'].iloc[0]
    ax.set_title(f"{env}: K={num_arms}, N={num_cps}")
    ax.set_xlabel(r'$\log(1/\delta)$')
    ax.grid(True, which='both', alpha=0.3)

axes[0].set_ylabel('Mean budget (stopping time)')
axes[-1].legend(loc='best')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'paper_envs_budget_vs_delta.pdf')
plt.savefig(RESULTS_DIR / 'paper_envs_budget_vs_delta.png', dpi=150)
plt.show()


# ---------- Optional: success rate plot ----------
fig, axes = plt.subplots(1, len(environments),
                         figsize=(5 * len(environments), 4.5),
                         sharex=True, sharey=True)
if len(environments) == 1:
    axes = [axes]

for ax, env in zip(axes, environments):
    d_env = df_discrete[df_discrete['environment'] == env]
    c_env = df_continuous[df_continuous['environment'] == env]

    agg_d = d_env.groupby('log_inv_delta')['success'].mean()
    agg_c = c_env.groupby('log_inv_delta')['success'].mean()

    ax.plot(agg_d.index, agg_d.values,
            marker='o', label='Discrete (MCPI)', color='tab:blue')
    ax.plot(agg_c.index, agg_c.values,
            marker='s', label='Continuous', color='tab:red')

    ax.set_title(f"{env}")
    ax.set_xlabel(r'$\log(1/\delta)$')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, which='both', alpha=0.3)

axes[0].set_ylabel('Success rate')
axes[-1].legend(loc='best')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'paper_envs_success_vs_delta.pdf')
plt.savefig(RESULTS_DIR / 'paper_envs_success_vs_delta.png', dpi=150)
plt.show()