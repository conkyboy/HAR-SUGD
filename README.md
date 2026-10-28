````
# HAR-SUGD

Companion code for "Compact and Parameter-Free Attribute Reduction
via Symmetric Uncertainty and Knowledge Granularity" (submitted to
Knowledge-Based Systems).

HAR-SUGD performs attribute reduction via a forward-greedy search:
symmetric-uncertainty (SU) significance drives both selection and
stopping (tolerance ε = 1e-9), and knowledge granularity breaks
significance ties.

## Repository layout
```
src/      implementation, baselines, drivers, analysis, verification
folds/    fixed 10-fold assignments (seed 42), shared by every method
results/  archived per-fold results matching the paper
```

| `src/` file(s) | Role |
|---|---|
| `harsugd_core.py`, `load_dataset.py`, `baselines.py`, `fsrmi.py`, `fdiwfs.py` | core implementation and competing selectors |
| `run_experiments.py` | main experiment: all methods × 18 datasets × 10 folds |
| `run_fcbf.py`, `fcbf_report.py` | FCBF baseline and δ sensitivity sweep |
| `run_audit_ties.py`, `tie_report.py`, `verify_ties.py`, `verify_gd_split.py` | SU-tie audit and independent verification |
| `eps_check.py` | ε-invariance check |
| `analyze_final.py`, `stats_upgrade.py`, `make_tables.py` | merged statistics, Friedman/Nemenyi/Holm, tables |
| `su_curves.py`, `overview_figure.py` | figures |
| `validate_data.py`, `orl40_build.py` | dataset validation / construction |

## Environment
Python 3.13, scikit-learn 1.9.0, pandas 3.0.3, NumPy 2.4.5, SciPy
(see `requirements.txt`).

## Reproduction
All scripts are run **from the repository root** (imports resolve via
`src/`; the fixed folds in `folds/` are read relative to the root):

```bash
python src/run_experiments.py    # main experiment (Tables 3-8)
python src/run_fcbf.py           # FCBF baseline + δ sweep
python src/eps_check.py          # ε-invariance check
python src/run_audit_ties.py     # tie audit
```
Newly generated result files appear in the repository root;
`results/` holds the archived versions reported in the paper.
To re-run only the analysis on the archived results:

```bash
cd results && python ../src/analyze_final.py
```

## Data
The 18 datasets are publicly available from the
[UCI ML Repository](https://archive.ics.uci.edu) and the
[ASU feature-selection repository](https://featureselection.asu.edu)
(see Table 1 of the paper); place them locally following the paths
in `src/load_dataset.py`. Datasets are not redistributed here for
licensing reasons.

## Citation
Manuscript under review; citation to be added upon acceptance.
````