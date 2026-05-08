# Reviewer Reproducibility Notes

This document summarizes how to reproduce the experiments with the reduced reviewer package.

## Included submission scripts

The `scripts/` directory intentionally contains only four SLURM submission scripts:

```text
scripts/submit_part1_unified_all.sh
scripts/submit_part1_unified_no_outage_lag.sh
scripts/submit_part2_ensemble_all.sh
scripts/submit_part2_ensemble_no_outage_lag.sh
```

No monitoring helper scripts and no `autoregressive_only` or `weather_only` submission scripts are included in this package.

## Required data files

Place the externally archived data files in the repository-level `data/` directory:

```text
data/merged_NWS_lag1.parquet
data/merged_NWS_lag12.parquet
data/merged_NWS_lag24.parquet
```

The files should be downloaded from the data archive listed in the main README.

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Submit the included experiments

```bash
sbatch scripts/submit_part1_unified_all.sh
sbatch scripts/submit_part1_unified_no_outage_lag.sh
sbatch scripts/submit_part2_ensemble_all.sh
sbatch scripts/submit_part2_ensemble_no_outage_lag.sh
```

Each script launches six array tasks covering LightGBM and XGBoost at lag horizons 1, 12, and 24.

## Cluster-specific settings

The SLURM scripts include resource requests such as wall time, CPU count, memory, and array size. These should be edited if the reviewer is using a different computing environment.

The scripts do not include a personal email address or a hard-coded personal project path. They use paths relative to the repository root.

## Expected outputs

Outputs are written under:

```text
outputs/<MODEL_TYPE>/<ALGORITHM>/lag<LAG>/weather_<WEATHER_SOURCE>/features_<FEATURE_SET>/
```

For Ensemble runs, outputs are further separated by event type:

```text
outputs/Ensemble/<ALGORITHM>/lag<LAG>/weather_<WEATHER_SOURCE>/features_<FEATURE_SET>/EVENT_TYPE_<EVENT_TYPE>/
```

Generated output files are intentionally not tracked by Git.
