📂 Data Availability & Sharing

The model computations were performed using the Texas A&M High Performance Research Computing (HPRC) facility.

The parquet data files (lag1, lag12, and lag24) are not included in this repository at this stage due to data-sharing restrictions.
These files will only be uploaded if the associated research paper is accepted.

Upon paper acceptance:

The dataset files will be uploaded to the data/ directory in this repository.

The GitHub repository will be linked to Zenodo, enabling DOI generation and long-term data preservation.

The data will be shared under appropriate licensing and citation requirements.

Until then, this repository provides the complete model training framework, scripts, and reproducible workflow—except for the raw data files.

🚀 How to Run the Analysis (Once Data is Available)

After the parquet files are uploaded to the data/ directory, the analysis can be executed using:

python train_optimized_event_id.py data/lag1.parquet Unified LightGBM 1 outputs/ --resume
python train_optimized_event_id.py data/lag12.parquet Ensemble XGBoost 12 outputs/ --resume
python train_optimized_event_id.py data/lag24.parquet Unified LightGBM 24 outputs/ --resume

🔐 Data Access Policy

⚠️ The original dataset is currently unavailable in this repository.
It will be released publicly only if the paper is accepted, with proper licensing through Zenodo.

🔗 Planned Open Science Integration
Component	Status	Plan
Parquet Data Files	❌ Not Public	Will be uploaded upon paper acceptance
GitHub–Zenodo Link	🕒 Pending	Will be activated for DOI and citation
Code & Scripts	✅ Available	Fully included in this repository
Reproducibility	⚠️ Partial	Fully reproducible once data is released
📌 Notes

The repository already includes the complete training architecture, Bayesian optimization, ensemble mechanism, and permutation importance computation.

The code is fully compatible with the uploaded data once released.

The repository aims to support reproducibility and transparency for scientific publication.
