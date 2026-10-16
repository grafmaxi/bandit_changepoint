# The Sample Complexity of Multiple Change Point Identification under Bandit Feedback

Code for the NeurIPS 2026 paper

> **The Sample Complexity of Multiple Change Point Identification under Bandit Feedback**
> Maximilian Graf\* (Universität Potsdam) and Victor Thuot\* (INRAE, Institut Agro, Univ Montpellier)
> *40th Conference on Neural Information Processing Systems (NeurIPS 2026)*
> \* Equal contribution.
>
> Paper: *link coming soon*

## Overview

We study multiple change point localization under bandit feedback. An unknown piecewise-constant function on `[0, 1]` can be queried sequentially at adaptively chosen inputs, and each query returns a noisy evaluation. The goal is to identify `N` change points to precision `η` with confidence `1 − δ`, using as few samples as possible.

This repository implements **LCP** (`LocalizeChangePoints`, Algorithm 2 in the paper). It also contains the experiments that compare LCP to the Track-and-Stop baselines **MCPI** and **SCPI** of Lazzaro and Pike-Burke (2025).

LCP follows a doubling schedule. At each stage it runs four steps:

1. **DetectIntervals** (Algorithm 1): multiscale tests on dyadic grids find intervals that contain a change point.
2. **EstimateJumps** (Algorithm 3): adaptively estimates the jump size on each detected interval.
3. **SHB** (Algorithm 4): sequential halving with backtracking localizes each change point to precision `η`.
4. **VerifyCP** (Algorithm 5): a two-sample test certifies each estimate.

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from src.models.continuous_cp_bandit import Continuous_Bandit
from src.algorithms.localize_change_points import localize_change_points

# Piecewise-constant function on [0, 1] with change points at 0.3 and 0.55, Gaussian noise (sd 1)
bandit = Continuous_Bandit(change_points=[0.3, 0.55], values=[0, 1, 0],
                           sampling_type="gaussian", parameter=1)

cps, budget, *phase_budgets = localize_change_points(
    bandit, num_cps=2, delta=0.05, delta_explore=1, precision=2**-8)
```

`phase_budgets` holds the samples spent in the detection, estimation, localization and verification phases.

## Implementation details

All experimental results in the paper were produced with the code in this repository. This implementation differs from the pseudocode in the paper (Algorithms 2–5) in a few minor points, mostly in the constants. The theoretical guarantees in the paper are proved for the pseudocode.

| Component | Paper | Implementation | Code |
|---|---|---|---|
| `EstimateJumps` (Alg. 3): acceptance threshold | $\sqrt{2^{-(k-5)}\log\left(\pi^2\lvert\mathcal{I}\rvert k^2/(3\delta)\right)}$ | $\sqrt{2^{-(k-3)}\log\left(\pi^2\lvert\mathcal{I}\rvert k^2/(3\delta)\right)}$ | [estimate_jumps.py:27](src/algorithms/estimate_jumps.py#L27) |
| `VerifyCP` (Alg. 5): threshold | $\sqrt{16\log(2/\delta)/T}$ | $\sqrt{32\log(2/\delta)/T}$ | [localize_change_points.py:83](src/algorithms/localize_change_points.py#L83) |
| `SHB` (Alg. 4): number of rounds | $\lceil 6\log(\lvert I\rvert/\eta)\rceil$ | $\lceil 6\log(1/\eta)\rceil$ | [sequential_halving_with_backtracking.py:49](src/algorithms/sequential_halving_with_backtracking.py#L49) |
| `LCP` (Alg. 2): first stage of the doubling schedule | $k=\lceil\log_2(2N)\rceil$ | $k=\lceil\log_2 N\rceil$ | [localize_change_points.py:25](src/algorithms/localize_change_points.py#L25) |
| `LCP` (Alg. 2): minimum budget per interval | $1$ | $2$ | [localize_change_points.py:45](src/algorithms/localize_change_points.py#L45) |

Two further differences concern `LCP` (Algorithm 2):

- **Interval order** ([localize_change_points.py:44](src/algorithms/localize_change_points.py#L44)): The $N$ intervals are taken in the order in which `EstimateJumps` accepts them, not sorted by their estimated jump (Line 7).
- **Short intervals** ([localize_change_points.py:50](src/algorithms/localize_change_points.py#L50)): For intervals of length at most $2\eta$, `SHB` returns the midpoint and the verification step (Line 10) is skipped.

## Reproducing the experiments

Each experiment has two scripts. An experiment script writes CSV results, and a plot script turns those results into a figure. Both write to `src/results_continuous/`. Run all commands from the repository root:

```bash
python -m src.experiments_continuous.<name>   # run Monte-Carlo simulation -> CSV
python -m src.plots_continuous.<name>         # CSV -> figure (.pdf / .png)
```

| Paper | Setting | `<name>` | Figure file | Runtime\* |
|---|---|---|---|---|
| Experiment 1, Fig. 1 | 2 CPs, varying spacing `s`, `η = 2⁻¹¹`, `δ = 0.05` | `two_change_points` | `two_change_points_budget.pdf` | ~1.9 h |
| Experiment 2, Fig. 2a | 2 CPs, `s = 1/4`, varying `δ`, `η = 2⁻⁸` | `varying_delta` | `varying_delta_budget.pdf` | ~17 min |
| Experiment 3, Fig. 2b | 2 CPs, `s = 1/4`, varying `η`, `δ = 0.05` | `varying_eta` | `varying_eta_budget.pdf` | ~27 min |
| Experiment 4, Fig. 3 | 1 CP, varying `δ`, `η = 2⁻⁷` | `one_change_point` | `one_change_point_budget.pdf` | ~10 min |
| Experiment 5, Fig. 4 | 10 CPs, varying `δ` and `η` | `many_change_points` | `many_cps_budget.pdf` | ~44 min |

\*Total runtimes from Table 2 of the paper. They were measured on a MacBook Air (Apple M1, 8 GB RAM) with 4 worker processes.

All settings are defined in the configuration block at the top of each experiment script: number of Monte-Carlo runs, `δ`, `η`, `δ_explore` and `num_workers`. Run `i` is seeded with `np.random.seed(i)`. The CSV results behind the paper figures are included in `src/results_continuous/`, so you can regenerate the plots without rerunning the simulations.

MCPI and SCPI are designed for discrete bandits. They therefore run on a discretization of the continuous problem with `⌊1/η⌋ + 1` arms (see Appendix F of the paper).

## Tests

```bash
python -m pytest src/tests
```

## Repository structure

```
src/
├── algorithms/               # LCP and its subroutines, plus the MCPI/SCPI baselines
├── models/                   # Bandit environments (continuous and discrete)
├── experiments_continuous/   # Experiment scripts for the paper figures
├── plots_continuous/         # Plot scripts for the paper figures
├── results_continuous/       # CSV results and figures
└── tests/
```

## Citation

The BibTeX entry will be completed once the proceedings are published.

```bibtex
@inproceedings{graf2026sample,
  title     = {The Sample Complexity of Multiple Change Point Identification under Bandit Feedback},
  author    = {Graf, Maximilian and Thuot, Victor},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
