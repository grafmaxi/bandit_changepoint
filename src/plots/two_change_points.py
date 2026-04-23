"""
Plot mean budgets against k for discrete and continuous methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
# ---------- Load data ----------
df_discrete = pd.read_csv(RESULTS_DIR / "results_discrete.csv")
df_continuous = pd.read_csv(RESULTS_DIR / "results_continuous.csv")

# ---------- Aggregate ----------
agg_discrete = df_discrete.groupby('gap')['budget'].agg(
    mean='mean',
    q05=lambda x: np.quantile(x, 0.05),
    q95=lambda x: np.quantile(x, 0.95),
)
agg_continuous = df_continuous.groupby('gap')['budget'].agg(
    mean='mean',
    q05=lambda x: np.quantile(x, 0.05),
    q95=lambda x: np.quantile(x, 0.95),
)

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(agg_discrete.index, agg_discrete['mean'],
        marker='o', label='Discrete (MCPI)', color='tab:blue')
ax.fill_between(agg_discrete.index, agg_discrete['q05'], agg_discrete['q95'],
                alpha=0.2, color='tab:blue')

ax.plot(agg_continuous.index, agg_continuous['mean'],
        marker='s', label='Continuous (localize_change_points)', color='tab:red')
ax.fill_between(agg_continuous.index, agg_continuous['q05'], agg_continuous['q95'],
                alpha=0.2, color='tab:red')

ax.set_xlabel('length of the plateau ($s\cdot m$)')
ax.set_ylabel('Mean budget')
ax.set_yscale('log')
ax.set_title('Mean sample complexity vs. $s\cdot m$')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'budget_vs_k.pdf')
plt.savefig(RESULTS_DIR / 'budget_vs_k.png', dpi=150)
plt.show()