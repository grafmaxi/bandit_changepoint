"""
Figure 1: plot mean budget against plateau size s for the two-change-points
experiment, comparing LCP with MCPI.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_continuous"

df_discrete = pd.read_csv(RESULTS_DIR / "results_discrete.csv")
df_continuous = pd.read_csv(RESULTS_DIR / "results_continuous.csv")


fig, ax = plt.subplots(figsize=(3, 1.8))
agg_d = df_discrete.groupby('s')['budget'].agg(
    mean='mean',
    q05=lambda v: np.quantile(v, 0.05),
    q95=lambda v: np.quantile(v, 0.95),
)
agg_c = df_continuous.groupby('s')['budget'].agg(
    mean='mean',
    q05=lambda v: np.quantile(v, 0.05),
    q95=lambda v: np.quantile(v, 0.95),
)

ax.plot(agg_c.index, agg_c['mean'], marker='s', label='LCP', color='tab:red')
ax.fill_between(agg_c.index, agg_c['q05'], agg_c['q95'], alpha=0.2, color='tab:red')

ax.plot(agg_d.index, agg_d['mean'], marker='o', label='MCPI', color='tab:blue')
ax.fill_between(agg_d.index, agg_d['q05'], agg_d['q95'], alpha=0.2, color='tab:blue')



# Reference line: 1/s, scaled to match the data range
s_values = np.array(sorted(agg_d.index))
ref_constant = 4*10**3
ax.plot(s_values, ref_constant / s_values,
        linestyle=':', color='orange', label=r'$\propto 1/s$')

ax.set_xlabel(r'plateau size $s$')
ax.set_ylabel('Mean budget')
ax.legend(loc = 'upper right')
ax.grid(True, which='both', alpha=0.3)
plt.rcParams.update({
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'two_change_points_budget.pdf', pad_inches = 0.2)
plt.savefig(RESULTS_DIR / 'two_change_points_budget.png', dpi=150, pad_inches = 0.2)
plt.show()

