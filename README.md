# ⚡ Outage Prediction using Weather-Based Lagged Features  
**Machine Learning Framework with EVENT_TYPE-Specific and Multi-Seed Ensemble Modeling**

This repository presents a machine learning framework for outage prediction in **Texas, USA**, utilizing **weather-based lagged features** and **event-specific modeling** derived from the **National Weather Service (NWS) Storm Events Database**.  
The framework trains **separate predictive models for each weather-driven outage event type**, enhancing interpretability and improving prediction accuracy compared to traditional global models.  
In addition, **Multi-Seed Ensemble Learning** is employed to improve statistical robustness, reduce variance, and support uncertainty-aware forecasting.

---

## 1. EVENT_TYPE-Specific Modeling (10 NWS Categories)

The framework independently models each of the following weather-induced outage event types, consistent with NWS storm event classification:

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

Each model is trained using tailored feature sets and lagged weather variables pertinent to the corresponding event mechanism.

---

## 2. Multi-Seed Ensemble Learning

To enhance stability and reduce randomness in model outputs, multiple training runs with different random seeds are executed. Predictions from these runs are aggregated (mean, median, or voting), resulting in:

- Reduced variance in model performance  
- Improved resistance to overfitting  
- Enhanced reliability for event-based outage forecasting  

---

## 3. Framework Components

- Event duration extraction based on NWS start and end timestamps  
- Lagged weather feature construction (1 hr, 12 hr, 24 hr)  
- Model training with LightGBM and XGBoost  
- Bayesian hyperparameter optimization  
- Permutation-based feature importance  
- Checkpointing and resume functionality  
- HPC optimization for Texas A&M HPRC (Slurm-based execution)

---

## 4. Weather and Event Data

- EVENT_TYPE labeling and duration follow the **NWS Storm Events Database** standards.  
- Weather variables include wind gust, precipitation, temperature, snowfall, ice accretion, soil moisture, freezing rain, humidity, and related conditions.  
- Geographic alignment is based on **Texas utility service territories**, ensuring practical relevance for electric grid operations.

---

## 5. Data Availability

The parquet files (`lag1`, `lag12`, `lag24`) are not included due to institutional and publication restrictions.

Upon acceptance of the associated research article, the following will be released:

- Data files in the `/data` directory  
- Archived dataset via Zenodo with DOI assignment  
- Licensing and citation instructions  

Until then, the repository provides the complete modeling code, workflow, and reproducible training pipeline, excluding raw datasets.

---

## 6. Execution

### 6.1 Local or Command-Line Execution

```bash
python train_optimized_final.py data/lag1.parquet Ensemble LightGBM 1 outputs/ --resume
python train_optimized_final.py data/lag12.parquet Ensemble XGBoost 12 outputs/ --resume
python train_optimized_final.py data/lag24.parquet Ensemble LightGBM 24 outputs/ --resume

### 6.2 HPC Execution (TAMU HPRC, Slurm)

Submit batch jobs using the following Slurm commands:

```bash
sbatch submit_part1_unified_improved.sh
sbatch submit_part2_ensemble_improved.sh
