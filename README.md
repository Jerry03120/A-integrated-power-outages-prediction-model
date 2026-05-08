# Outage Prediction ML Experiments

This repository contains the code package for the machine-learning experiments associated with the manuscript under review.

The package is organized so that the code is stored in GitHub, while the input data files are archived separately through Zenodo, DesignSafe, or a similar research-data repository.

## Repository structure

```text
outage_prediction_ml_github/
├── README.md
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── LICENSE
├── data/
│   └── README.md
├── outputs/
│   └── README.md
├── docs/
│   └── reviewer_reproducibility.md
├── src/
│   └── train_optimized_hpc_singlemode.py
└── scripts/
    ├── submit_part1_unified_all.sh
    ├── submit_part1_unified_no_outage_lag.sh
    ├── submit_part2_ensemble_all.sh
    └── submit_part2_ensemble_no_outage_lag.sh
```

Only the four SLURM submission scripts listed above are included in this reviewer package.

## Data availability

The input data files are not included in this GitHub repository. After downloading the dataset from Zenodo, DesignSafe, or another archival repository, place the files in the `data/` directory at the repository root.

Expected files:

```text
data/merged_NWS_lag1.parquet
data/merged_NWS_lag12.parquet
data/merged_NWS_lag24.parquet
```

Update the following placeholders after the data archive is finalized:

- Zenodo DOI/link: `[insert DOI or URL]`
- DesignSafe DOI/link: `[insert DOI or URL]`

## Installation

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import pandas, numpy, lightgbm, xgboost, sklearn, skopt, pyarrow, scipy; print('OK')"
```

## Quick local test

A lightweight test can be run with permutation importance disabled:

```bash
python src/train_optimized_hpc_singlemode.py \
  data/merged_NWS_lag1.parquet \
  Unified \
  LightGBM \
  1 \
  outputs \
  --resume \
  --feature-set all \
  --weather-source lagged_observed \
  --no-importance
```

## SLURM submission scripts

This package includes four submission scripts:

```text
scripts/submit_part1_unified_all.sh
scripts/submit_part1_unified_no_outage_lag.sh
scripts/submit_part2_ensemble_all.sh
scripts/submit_part2_ensemble_no_outage_lag.sh
```

Each script launches a 6-task SLURM array:

1. LightGBM, lag 1
2. LightGBM, lag 12
3. LightGBM, lag 24
4. XGBoost, lag 1
5. XGBoost, lag 12
6. XGBoost, lag 24

Submit the full set with:

```bash
sbatch scripts/submit_part1_unified_all.sh
sbatch scripts/submit_part1_unified_no_outage_lag.sh
sbatch scripts/submit_part2_ensemble_all.sh
sbatch scripts/submit_part2_ensemble_no_outage_lag.sh
```

The scripts use paths relative to the repository root. They also assume a virtual environment at `${JOB_ROOT}/.venv` by default. You can override this at submission time:

```bash
VENV_PATH=/path/to/your/venv sbatch scripts/submit_part1_unified_all.sh
```

The SLURM resource requests are intentionally written as editable reviewer defaults. Please adjust time, CPU, memory, and other scheduler options as needed for your cluster.

## Output structure

Generated results are written to `outputs/`, which is excluded from Git tracking.

Example output directories:

```text
outputs/Unified/LightGBM/lag1/weather_lagged_observed/features_all/
outputs/Unified/XGBoost/lag24/weather_lagged_observed/features_no_outage_lag/
outputs/Ensemble/LightGBM/lag12/weather_lagged_observed/features_all/EVENT_TYPE_Storm/
outputs/Ensemble/XGBoost/lag24/weather_lagged_observed/features_no_outage_lag/EVENT_TYPE_Storm/
```

## Reproducibility notes

The main training script supports:

- Unified and Ensemble modeling modes
- LightGBM and XGBoost
- lag horizons of 1, 12, and 24
- feature configurations included in this package: `all` and `no_outage_lag`
- lagged observed weather features by default
- optional forecast weather features through `WEATHER_SOURCE_MODE=forecast`
- forward-in-time EVENT_ID cross-validation
- naive persistence baseline comparison
- grouped permutation importance

For detailed reviewer instructions, see `docs/reviewer_reproducibility.md`.
