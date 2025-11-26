# ⚡ Outage Prediction using Weather-Based Lagged Features  
Machine Learning Framework with Unified and Ensemble Modeling

This repository contains a complete framework for training predictive models using weather-based lagged features for outage forecasting.  
It includes **Unified** (single global model) and **Ensemble** (per EVENT_TYPE) modeling strategies using **LightGBM** and **XGBoost**, with support for:

✔ Bayesian Optimization  
✔ Seed-based Ensemble Predictions  
✔ Permutation Importance  
✔ Checkpointing & Resume  
✔ HPC-Optimized Training (TAMU HPRC)

---

## 📂 Data Availability & Sharing

The model computations were performed using the  
**[Texas A&M High Performance Research Computing (HPRC)](https://hprc.tamu.edu/)** facility.

Due to data-sharing restrictions, the parquet files (`lag1`, `lag12`, `lag24`) are **not currently included** in this repository.

These datasets **will only be released if the associated research paper is accepted.**

### Upon Paper Acceptance:
✔ Datasets will be uploaded to the `data/` directory  
✔ Repository will be linked to **Zenodo for DOI generation & archival**  
✔ Data will be shared under a proper license and citation format  

Until then, this repository contains the **complete model training code and reproducible workflow**, excluding raw data files.

---

## 🚀 How to Run the Analysis (Once Data Is Available)

After uploading parquet files to `data/`, run the training:

```bash
python train_optimized.py data/lag1.parquet Unified LightGBM 1 outputs/ --resume
python train_optimized.py data/lag12.parquet Ensemble XGBoost 12 outputs/ --resume
python train_optimized.py data/lag24.parquet Unified LightGBM 24 outputs/ --resume

or

sbatch submit_part1_unified_improved.sh
sbatch submit_part2_ensemble_improved.sh
