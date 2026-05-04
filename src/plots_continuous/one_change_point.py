"""
Plot mean budget and deviation against log(1/delta) for the one-change-point
experiment, comparing discrete (CPI) and continuous methods.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_continuous"

df_discrete = pd.read_csv(RESULTS_DIR / "one_change_point_results_discrete.csv")
df_continuous = pd.read_csv(RESULTS_DIR / "one_change_point_results_continuous.csv")


def aggregate(df, x):
    return df.groupby(x)['budget'].agg(
        mean='mean',
        q05=lambda v: np.quantile(v, 0.05),
        q95=lambda v: np.quantile(v, 0.95),
    ), df.groupby(x)['deviation'].mean()


# ---------- Budget ----------
fig, ax = plt.subplots(figsize=(5, 3))
agg_d, _ = aggregate(df_discrete, 'log_inv_delta')
agg_c, _ = aggregate(df_continuous, 'log_inv_delta')



ax.plot(agg_c.index, agg_c['mean'], marker='s', label='LCP', color='tab:red')
ax.fill_between(agg_c.index, agg_c['q05'], agg_c['q95'], alpha=0.2, color='tab:red')

ax.plot(agg_d.index, agg_d['mean'], marker='o', label='MCPI', color='tab:blue')
ax.fill_between(agg_d.index, agg_d['q05'], agg_d['q95'], alpha=0.2, color='tab:blue')

det_c = df_continuous.groupby('log_inv_delta')['detection_budget'].mean()
#ax.plot(det_c.index, det_c.values, marker='^', linestyle='--',
#        label='LCPI (DI only)', color='tab:red', alpha=0.7)

ax.set_xlabel(r'$\log(1/\delta)$')
ax.set_ylabel('Mean budget')
#ax.set_title('One change point: budget vs. confidence level')
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
plt.savefig(RESULTS_DIR / 'one_change_point_budget.pdf', pad_inches=0.2)
plt.savefig(RESULTS_DIR / 'one_change_point_budget.png', dpi=150, pad_inches=0.2)
plt.show()
