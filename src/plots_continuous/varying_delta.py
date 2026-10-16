"""
Figure 2a: plot mean budget against log(1/delta) for the varying-delta
experiment, comparing LCP with MCPI.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_continuous"

df_discrete = pd.read_csv(RESULTS_DIR / "varying_delta_results_discrete.csv")
df_continuous = pd.read_csv(RESULTS_DIR / "varying_delta_results_continuous.csv")


fig, ax = plt.subplots(figsize=(3, 2.3))
agg_d = df_discrete.groupby('log_inv_delta')['budget'].agg(
    mean='mean',
    q05=lambda v: np.quantile(v, 0.05),
    q95=lambda v: np.quantile(v, 0.95),
)
agg_c = df_continuous.groupby('log_inv_delta')['budget'].agg(
    mean='mean',
    q05=lambda v: np.quantile(v, 0.05),
    q95=lambda v: np.quantile(v, 0.95),
)

ax.plot(agg_c.index, agg_c['mean'], marker='s', label='LCP', color='tab:red')
ax.fill_between(agg_c.index, agg_c['q05'], agg_c['q95'], alpha=0.2, color='tab:red')

ax.plot(agg_d.index, agg_d['mean'], marker='o', label='MCPI', color='tab:blue')
ax.fill_between(agg_d.index, agg_d['q05'], agg_d['q95'], alpha=0.2, color='tab:blue')


ax.set_xlabel(r'$\log(1/\delta)$')
ax.set_ylabel('Mean budget')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.rcParams.update({
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})
plt.savefig(RESULTS_DIR / 'varying_delta_budget.pdf', bbox_inches = "tight", pad_inches = 0.2)
plt.savefig(RESULTS_DIR / 'varying_delta_budget.png', dpi=150, bbox_inches = "tight", pad_inches = 0.2)
plt.show()

