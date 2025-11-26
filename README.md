# ⚡ Outage Prediction using Weather-Based Lagged Features  
**Machine Learning Framework with EVENT_TYPE-Specific & Multi-Seed Ensemble Modeling**

This repository introduces a high-performance machine learning framework for **power outage prediction in Texas, USA**, using **NWS-defined EVENT_TYPE classifications** and **weather-based lagged features**.  
Unlike conventional global models, this framework builds **independent predictive models for each event type**, enabling **event-specific learning**, higher interpretability, and improved model accuracy.  
It further enhances reliability via **Multi-Seed Ensemble Learning**, offering robust and uncertainty-aware predictions.

---

## 🌪️ EVENT_TYPE-Specific Modeling (10 NWS Categories)

This framework trains **distinct models** for each of the following **10 major NWS storm event types**:

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

Each model learns **unique outage-causing behavior** based on the characteristics of the specific event type.

---

## 🎯 Multi-Seed Ensemble Learning

To improve **predictive stability, uncertainty estimation**, and **generalization**, the framework uses **seed-based ensemble modeling**, where multiple model instances (with different random seeds) are trained and aggregated.

✔ Reduces random variability  
✔ Improves resilience to overfitting  
✔ Enables probabilistic and confidence-aware forecasting  
✔ Ensemble averaging / median / voting supported  

---

## 🚀 Framework Capabilities

- NWS-Based EVENT_TYPE & Duration Extraction  
- Lagged Weather Feature Engineering (1h, 12h, 24h)  
- LightGBM & XGBoost Support  
- Bayesian Hyperparameter Optimization  
- Permutation-Based Importance  
- Checkpointing & Resume Support  
- HPC Optimization for **Texas A&M HPRC (Slurm)**

---

## 🌤️ Weather & Event Data Source

- EVENT_TYPE labels follow the **National Weather Service (NWS) Storm Events Database** standards.  
- Event duration is calculated using **start and end timestamps** from NWS-reported events.  
- Weather features include wind gust, temperature, precipitation, humidity, soil moisture, freezing rain, ice accumulation, snowfall, etc.  
- The dataset is aligned with **Texas utility service regions**, making the prediction geographically relevant and operationally meaningful.

---

## 📂 Data Availability

⚠️ Due to institutional and publication restrictions, the parquet files (`lag1`, `lag12`, `lag24`) are **not included** in this repository.

📌 **Upon acceptance of the research paper:**

✔ Data files will be uploaded to the `data/` directory  
✔ The dataset will be archived via **Zenodo with DOI assignment**  
✔ Licensing, citation format, and metadata will be published  

Until then, all **training code, modeling logic, and workflows** are fully included for reproducibility — except raw datasets.

---

## 🚀 How to Run the Framework

### ▶️ Local / Terminal Execution

```bash
python train_optimized_final.py data/lag1.parquet Ensemble LightGBM 1 outputs/ --resume
python train_optimized_final.py data/lag12.parquet Ensemble XGBoost 12 outputs/ --resume
python train_optimized_final.py data/lag24.parquet Ensemble LightGBM 24 outputs/ --resume

### 💻 HPC Execution on TAMU HPRC (Slurm)

Submit batch training using the provided Slurm scripts:

```bash
sbatch submit_part1_unified_improved.sh
sbatch submit_part2_ensemble_improved.sh

