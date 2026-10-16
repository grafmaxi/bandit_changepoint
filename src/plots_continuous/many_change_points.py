"""
Figure 4: plot mean budget against log(1/delta) for the many-change-points
experiment, with one curve per (precision, method) combination.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_continuous"

df_discrete = pd.read_csv(RESULTS_DIR / "many_cps_results_discrete.csv")
df_continuous = pd.read_csv(RESULTS_DIR / "many_cps_results_continuous.csv")
fig, ax = plt.subplots(figsize=(5, 3))
precisions = sorted(df_discrete['precision'].unique(), reverse=True)
cmap = plt.cm.viridis
colors = [cmap(i / max(len(precisions) - 1, 1)) for i in range(len(precisions))]

continuous_handles = []
discrete_handles = []
total_lcp = df_continuous['runtime'].sum() if 'runtime' in df_continuous.columns else 0
total_mcpi = df_discrete['runtime'].sum() if 'runtime' in df_discrete.columns else 0
total_combined = total_lcp + total_mcpi

print(f"Total runtime LCP (sum across all runs):  {total_lcp:.2f} seconds "
      f"= {total_lcp/3600:.2f} hours")
print(f"Total runtime MCPI (sum across all runs): {total_mcpi:.2f} seconds "
      f"= {total_mcpi/3600:.2f} hours")
print(f"Total combined runtime:                   {total_combined:.2f} seconds "
      f"= {total_combined/3600:.2f} hours")


for prec, color in zip(precisions, colors):
    sub_d = df_discrete[df_discrete['precision'] == prec]
    sub_c = df_continuous[df_continuous['precision'] == prec]

    agg_d = sub_d.groupby('log_inv_delta')['budget'].mean()
    agg_c = sub_c.groupby('log_inv_delta')['budget'].mean()

    eta_label = fr'$\eta={prec:.3e}$'

    line_c, = ax.plot(
        agg_c.index, agg_c.values,
        marker='s', linestyle='--',
        label=f'LCP, {eta_label}',
        color=color,
    )
    line_d, = ax.plot(
        agg_d.index, agg_d.values,
        marker='o', linestyle='-',
        label=f'MCPI, {eta_label}',
        color=color,
    )

    continuous_handles.append(line_c)
    discrete_handles.append(line_d)

# Build legend with continuous on left column, discrete on right column
ordered_handles = continuous_handles + discrete_handles
ordered_labels = [h.get_label() for h in ordered_handles]


ax.set_xlabel(r'$\log(1/\delta)$')
ax.set_ylabel('Mean budget')
ax.legend(
    ordered_handles, ordered_labels,
    ncol=2,
    loc = 'upper center',
    bbox_to_anchor=(0.5, 0.67)
)
plt.rcParams.update({
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})



ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'many_cps_budget.pdf', bbox_inches = "tight", pad_inches = 0.02)
plt.savefig(RESULTS_DIR / 'many_cps_budget.png', dpi=150, bbox_inches = "tight", pad_inches = 0.02)
plt.show()