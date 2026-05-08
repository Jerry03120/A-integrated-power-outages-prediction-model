# Outage Prediction ML Experiments

This repository contains the machine-learning code used for the outage prediction experiments reported in the manuscript under review.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── src/
│   └── train_optimized_hpc_singlemode.py
├── scripts/
│   ├── submit_part1_unified_all.sh
│   ├── submit_part1_unified_autoregressive_only.sh
│   ├── submit_part1_unified_weather_only.sh
│   ├── submit_part1_unified_no_outage_lag.sh
│   ├── submit_part2_ensemble_all.sh
│   ├── submit_part2_ensemble_autoregressive_only.sh
│   ├── submit_part2_ensemble_weather_only.sh
│   ├── submit_part2_ensemble_no_outage_lag.sh
│   ├── monitor_progress_detailed.sh
│   └── check_specific_job.sh
└── outputs/
    └── README.md
```

## Data availability

The input data are archived separately and are not included in this GitHub repository. After downloading the dataset from Zenodo or DesignSafe, place the files in the `data/` directory:

```text
data/merged_NWS_lag1.parquet
data/merged_NWS_lag12.parquet
data/merged_NWS_lag24.parquet
```

Update `data/README.md` with the final Zenodo or DesignSafe DOI after the dataset has been deposited.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "import pandas, numpy, lightgbm, xgboost, sklearn, skopt, pyarrow, scipy; print('OK')"
```

## Quick local test

Use `--no-importance` for a faster test run.

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

## Full SLURM runs

The submit scripts are written so that the repository root is detected automatically. If needed, override the root directory or virtual environment at submission time:

```bash
JOB_ROOT=/path/to/repo VENV_PATH=/path/to/venv sbatch scripts/submit_part1_unified_all.sh
```

Unified model runs:

```bash
sbatch scripts/submit_part1_unified_all.sh
sbatch scripts/submit_part1_unified_no_outage_lag.sh
```

Ensemble model runs:

```bash
sbatch scripts/submit_part2_ensemble_all.sh
sbatch scripts/submit_part2_ensemble_no_outage_lag.sh
```

Each submit script runs six SLURM array tasks: LightGBM and XGBoost at lag horizons 1, 12, and 24.

## Optional weather-source mode

By default, the scripts use lagged observed weather features. To use forecast weather features:

```bash
WEATHER_SOURCE_MODE=forecast sbatch scripts/submit_part1_unified_all.sh
```

## Monitoring

Overall progress:

```bash
bash scripts/monitor_progress_detailed.sh
watch -n 30 bash scripts/monitor_progress_detailed.sh
```

Specific configuration:

```bash
bash scripts/check_specific_job.sh Unified LightGBM 1 all
bash scripts/check_specific_job.sh Ensemble XGBoost 24 no_outage_lag lagged_observed Storm
```

## Generated outputs

Outputs are written under `outputs/` and are not tracked by Git. Example paths:

```text
outputs/Unified/LightGBM/lag1/weather_lagged_observed/features_all/
outputs/Ensemble/XGBoost/lag24/weather_lagged_observed/features_no_outage_lag/EVENT_TYPE_Storm/
```

## Notes for reviewers

1. Download the archived data and place the three parquet files in `data/`.
2. Install dependencies from `requirements.txt`.
3. Run either the quick local command or the SLURM scripts.
4. Generated logs, checkpoints, and model outputs are written to `logs/` and `outputs/`.

## Citation

Please cite the associated manuscript and dataset DOI when using this code or data.
