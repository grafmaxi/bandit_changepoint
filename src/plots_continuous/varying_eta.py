"""
Plot mean budget and deviation against precision for the varying-eta
experiment, comparing discrete (MCPI) and continuous methods.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_continuous"

df_discrete = pd.read_csv(RESULTS_DIR / "varying_eta_results_discrete.csv")
df_continuous = pd.read_csv(RESULTS_DIR / "varying_eta_results_continuous.csv")


# ---------- Budget ----------
agg_c = df_continuous.groupby('precision')['budget'].agg(
    mean='mean',
    q05=lambda v: np.quantile(v, 0.05),
    q95=lambda v: np.quantile(v, 0.95),
)
fig, ax = plt.subplots(figsize=(3, 2.3))
agg_d = df_discrete.groupby('precision')['budget'].agg(
    mean='mean',
    q05=lambda v: np.quantile(v, 0.05),
    q95=lambda v: np.quantile(v, 0.95),
)

ax.plot(agg_c.index, agg_c['mean'], marker='s', label='LCP', color='tab:red')
ax.fill_between(agg_c.index, agg_c['q05'], agg_c['q95'], alpha=0.2, color='tab:red')

ax.plot(agg_d.index, agg_d['mean'], marker='o', label='MCPI', color='tab:blue')
ax.fill_between(agg_d.index, agg_d['q05'], agg_d['q95'], alpha=0.2, color='tab:blue')


# Reference line: 1/eta, anchored at the smallest precision (right side after invert)
eta_values = np.array(sorted(agg_d.index))
ref_constant = 2.5*10**2  # anchor at smallest eta
ax.plot(eta_values, ref_constant / eta_values,
        linestyle=':', color='orange', label=r'$\propto 1/\eta$')

ax.set_xlabel(r'precision $\eta$')
ax.set_ylabel('Mean budget')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.invert_xaxis()  # finer precision (smaller eta) on the right
#ax.set_title('Varying precision: budget vs. precision')
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
plt.savefig(RESULTS_DIR / 'varying_eta_budget.pdf', bbox_inches = "tight", pad_inches = 0.2)
plt.savefig(RESULTS_DIR / 'varying_eta_budget.png', dpi=150, bbox_inches = "tight", pad_inches = 0.2)
plt.show()