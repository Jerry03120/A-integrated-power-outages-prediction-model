# 🌐 Model Training Framework for Outage Prediction

This repository contains a complete and modular framework for training predictive models using lagged weather-based features, with support for Bayesian optimization, ensemble learning, and permutation importance. The computation was performed using the **[Texas A&M High Performance Research Computing (HPRC)](https://hprc.tamu.edu/)** facility.

---

## 📂 Data Availability & Sharing

The model computations utilized parquet data files (`lag1`, `lag12`, and `lag24`).  
Due to data-sharing restrictions, these files are **not included** in this repository at present.

These datasets will **only be released if the associated research paper is accepted**.

### Upon Paper Acceptance:
✔ The dataset files will be uploaded to the `data/` directory of this repository  
✔ The GitHub repository will be **linked to Zenodo** for DOI generation and archival  
✔ Data will be shared under appropriate **licensing and citation** requirements  

Until then, this repository provides the full reproducible codebase and model training pipeline — excluding the raw data files.

---

## 🚀 How to Run the Analysis (Once Data is Available)

After uploading the parquet dataset files to the `data/` directory, you can execute the model training pipeline using:

```bash
python train_optimized_event_id.py data/lag1.parquet Unified LightGBM 1 outputs/ --resume
python train_optimized_event_id.py data/lag12.parquet Ensemble XGBoost 12 outputs/ --resume
python train_optimized_event_id.py data/lag24.parquet Unified LightGBM 24 outputs/ --resume
