# ⚡ Outage Prediction using Weather-Based Lagged Features  
**A Scalable Machine Learning Framework with EVENT_TYPE-Specific and Multi-Seed Ensemble Modeling**

This repository provides a reproducible and high-fidelity machine learning framework for **weather-induced power outage prediction in Texas, USA**.  
Leveraging **meteorologically derived lagged features** and **event-specific modeling aligned with the National Weather Service (NWS) Storm Events Database**, the framework builds **distinct predictive models for each weather hazard type**, rather than relying on a single global model.  
This design enhances **physical interpretability, predictive precision, and model generalizability under heterogeneous weather conditions**.  
Additionally, **Multi-Seed Ensemble Learning** is incorporated to improve statistical reliability, reduce variance, and enable uncertainty-aware forecasting—supporting real-world resilience planning.

---

## 1. EVENT_TYPE-Specific Modeling (Aligned with NWS Hazard Taxonomy)

The framework independently models outage impacts across ten major weather-driven hazard types, following the official **NWS Storm Events Classification**:

| EVENT_TYPE |
|------------|
| Thunderstorm Winds |
| Winter Storm |
| Hail |
| Extreme Cold/Wind Chill |
| Flash Flood |
| Drought |
| Winter Weather |
| Ice Storm |
| Cold/Wind Chill |
| Frost/Freeze |

Each model is trained using tailored feature representations designed to capture hazard-specific outage mechanisms, including **wind-related damage, freezing precipitation, hydrological stress, and thermal extremes**.

---

## 2. Multi-Seed Ensemble Learning

To improve forecasting consistency and mitigate sensitivity to initialization randomness, the framework adopts a **multi-seed ensemble methodology**, whereby models are trained across multiple random seeds and aggregated using statistical consensus mechanisms:

- Variance reduction in performance metrics  
- Robustness against overfitting and single-seed instability  
- Support for deterministic deployment and confidence-aware inference  
- Ensemble-based aggregation methods (mean, median, majority voting)

This ensemble strategy aligns with best practices for **high-stakes forecasting applications** in infrastructure and risk-sensitive domains.

---

## 3. Framework Components

| Component | Description |
|-----------|-------------|
| Lagged Feature Engineering | Generation of meteorological lag features at 1h, 12h, and 24h intervals |
| Event Duration Modeling | Extraction of hazard duration using NWS event start/end timestamps |
| Learning Algorithms | LightGBM and XGBoost for classification and regression tasks |
| Optimization | Bayesian hyperparameter tuning for model generalization |
| Explainability | Permutation-based feature importance for interpretability |
| Resilience | Checkpointing, resume functionality, and seed-controlled execution |
| HPC Scalability | Native support for Slurm-based execution on TAMU HPRC |

---

## 4. Meteorological and Geospatial Data Considerations

- Event categories, severity, and temporal durations are sourced from the **NWS Storm Events Database**.  
- Weather feature engineering includes wind gust, precipitation intensity, snow/ice accumulation, air temperature, soil moisture, humidity, and freezing rain.  
- Modeling is geographically aligned with **Texas utility service regions**, incorporating spatially resolved grid outage data.

This ensures **physical relevance**, **geographic specificity**, and **operational applicability** of the forecasting outcomes.

---

## 5. Data Availability

Due to ongoing publication review and institutional restrictions, the foundational parquet files (`lag1`, `lag12`, `lag24`) are **not included** at this time.

Upon acceptance of the research article, the following will be released:

| Planned Release | Description |
|------------------|------------|
| `/data` Directory | Complete EVENT_TYPE-specific processed datasets |
| Zenodo DOI | Long-term archival with replicable workflow linkage |
| Licensing & Citation | Usage license and formal citation (IEEE/APA/BibTeX) |

Until then, this repository provides the complete reproducible pipeline, excluding raw data.

---

## 6. Execution

To facilitate reproducibility both locally and in HPC environments, unified command structures are provided.

```bash
#############################################
# Local or Command-Line Execution
#############################################

python train_optimized_final.py data/lag1.parquet  Ensemble  LightGBM  1   outputs/ --resume
python train_optimized_final.py data/lag12.parquet Ensemble  XGBoost   12  outputs/ --resume
python train_optimized_final.py data/lag24.parquet Ensemble  LightGBM  24  outputs/ --resume


#############################################
# HPC Execution (TAMU HPRC, Slurm)
#############################################

sbatch submit_part1_unified_improved.sh
sbatch submit_part2_ensemble_improved.sh
