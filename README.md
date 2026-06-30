# Explainable AI in Predicting and Analyzing Cardiovascular Disease Risk

Undergraduate thesis project exploring **Explainable AI (XAI)** for cardiovascular disease (CVD) risk modeling. The system trains and calibrates machine learning classifiers for two clinical scenarios, then uses **SHAP** to explain *which factors drove each prediction* and *by how much*:

- **Prognosis** - 10-year coronary heart disease (CHD) risk, using the **Framingham Heart Study** dataset.
- **Diagnosis** - presence of heart disease, using the **Cleveland Heart Disease (UCI)** dataset.

A Streamlit application (`app.py`) lets a user enter patient data (or pick a sample from the held-out test set), see the model's risk prediction, and inspect both **local** (SHAP waterfall/force plot) and **global** (SHAP beeswarm/bar plot) explanations.

| Scenario | Dataset | Target variable | Schema file | Model used by `app.py` |
|---|---|---|---|---|
| Prognosis | Framingham Heart Study | `TenYearCHD` | `config/fram.yaml` | `framingham/framingham_CatBoost_model.joblib` |
| Diagnosis | Cleveland Heart Disease (UCI) | `num` | `config/uci.yaml` | `cleveland/cleveland_CatBoost_model.joblib` |

---

## 1. Project Structure

```
cvd_7/
├── app.py                  # Streamlit app
├── main.py                 # End-to-end offline training pipeline 
├── config.py                # Loads the YAML column schemas below
├── config/
│   ├── fram.yaml             # Framingham column roles (numerical/binary/ordinal/target)
│   └── uci.yaml              # Cleveland column roles
├── loader.py                # CSV loading + stratified train/test split
├── preprocessing.py         # Imputation / encoding / scaling pipelines per model family
├── models.py                 # Candidate models + hyperparameter search grids
├── training.py               # Nested cross-validation + RandomizedSearchCV tuning
├── calibration.py            # Probability calibration (isotonic/sigmoid) + calibration curves
├── threshold_tuning.py       # Decision threshold optimization (Youden's J / F2-score)
├── utils.py                  # ThresholdClassifier wrapper, metrics, Decision Curve Analysis
├── shap_plot.py              # Generates the global/local SHAP plots used by app.py
├── shap_valid.py             # SHAP vs. ALE consistency + bootstrap SHAP stability checks
├── eda.ipynb                  # Exploratory data analysis notebook
├── uci_data_profile.html     # Data profile of Cleveland dataset generated from eda.ipynb
├── framingham_data_profile.html # Data profile of Framingham dataset generated from eda.ipynb
├── requirements.txt          # Python dependencies
├── cleveland/                 # Diagnosis artifacts: models, train/test splits, metrics, plots
│   ├── cleveland_CatBoost_model.joblib     # <- loaded by app.py
│   ├── cleveland_train_test_data.joblib    # <- loaded by app.py
│   └── ...                                  # other candidate models, CV/calibration/threshold CSVs & PNGs
├── framingham/                # Prognosis artifacts (same layout as cleveland/)
│   ├── framingham_CatBoost_model.joblib    # <- loaded by app.py
│   ├── framingham_train_test_data.joblib   # <- loaded by app.py
│   └── ...

```
---

## 2. Environment & Prerequisites

- **Python 3.10** (developed and tested on 3.10.11). CatBoost/XGBoost prebuilt wheels are most reliable on this minor version across Windows/macOS/Linux.
- **pip** for dependency installation.
- A few hundred MB of free disk space - the bundled model files (especially the Framingham CatBoost models) are large (see [Section 6](#6-notes)).

---

## 3. Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/khgthanhdinh/ITDSIU22161_PhamHongAn
   cd ITDSIU22161_PhamHongAn
   ```

2. **Create and activate a virtual environment**

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   macOS / Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Key libraries installed: `streamlit`, `streamlit-shap`, `shap`, `scikit-learn`, `catboost`, `xgboost`, `category_encoders`, `pandas`, `numpy`, `matplotlib`.

---

## 4. Running the App

> **Before this, see [Section 6](#6-notes) to find the missing file from `\framingham` folder.**

With the virtual environment active and from the project root:

```bash
streamlit run app.py
```

This opens the dashboard in your browser (default `http://localhost:8501`). In the UI:

1. **Sidebar → Choose Analysis Type**: `Prognosis (Framingham)` or `Diagnosis (Cleveland)`.
2. **Sidebar → Input Method**:
   - `Manual Entry` - fill in patient parameters using the sliders/dropdowns.
   - `Test Data Explorer` - pick a row index from the held-out test set; the app also shows the ground-truth label next to the model's prediction.
3. Click **Analyze Patient Data** (Prediction Result tab) to see:
   - The predicted risk class and probability vs. the model's tuned decision threshold.
   - A local SHAP **waterfall plot** and a interpretation of the top contributing factors.
   - A SHAP **force plot**.
4. **Global SHAP Analysis** tab - beeswarm and bar plots.
5. **Sample Dataset** tab - preview/download the training sample for the selected dataset.

No retraining is required to run the app - it loads the pretrained, calibrated CatBoost models and their corresponding train/test splits directly from the `.joblib` files in `cleveland/` and `framingham/` (see [Section 6](#6-notes) for Framingham case).

---

## 5. Retraining the Models (Optional)

> The `.joblib` files already in `cleveland/` and `framingham/` (find the file of Framingham in [Section 6](#6-notes)) are sufficient to run `app.py`. 

Retraining is **only needed** if you want to reproduce the full experimentation pipeline (model comparison, calibration, threshold tuning, SHAP plot generation) from raw data.

### 5.1 Get the raw data

`main.py` expects raw CSVs that are in:
```
./data/cleveland.csv
./data/framingham.csv
```

### 5.2 Run the training pipeline

```bash
python main.py
```

For each dataset, this will:
1. Split the data (stratified, `RANDOM_STATE=42`) and save it to `<dataset>/<dataset>_train_test_data.joblib`.
2. Run **nested cross-validation** over 6 candidate model families (Logistic, DecisionTree, SVM, RandomForest, XGBoost, CatBoost - see `models.py`), saving per-fold metrics CSVs.
3. **Hyperparameter-tune** the top-3 models from nested CV (plus Logistic Regression, always included) via `RandomizedSearchCV`.
4. **Calibrate** probabilities (`isotonic` for Framingham, `sigmoid/Platt` for Cleveland; Logistic Regression is left uncalibrated since it's natively well-calibrated) and save calibration curve plots.
5. **Tune the decision threshold** on out-of-fold predictions - Youden's J statistic (ROC curve) for Cleveland, F2-score (PR curve) for Framingham - and wrap each final model in the `ThresholdClassifier` from `utils.py`.
6. Save each final model as `<dataset>/<dataset>_<ModelName>_model.joblib`.

This can take a while, about **2h at least**, since it runs a randomized hyperparameter search with nested CV across multiple model families for both datasets, then finds best hyperparameter combinations again for each of the top-3 models, and many steps following.

### 5.3 Regenerate the SHAP plots used by the app

The "Global SHAP Analysis" tab reads static PNGs from the repo root. Regenerate them after retraining with:
```bash
python shap_plot.py
```
This produces `shap_<dataset>_global_bar.png`, `shap_<dataset>_global_beeswarm.png`, `shap_<dataset>_local_waterfall_idx0.png`, and `shap_<dataset>_local_force_idx0.png` for both datasets.

### 5.4 (Optional) SHAP/ALE validation & stability analysis

```bash
python shap_valid.py
```
Runs a SHAP-vs-ALE consistency check and a bootstrap SHAP stability analysis (Kendall's Tau, top-K feature overlap) for the CatBoost model, saving plots into `cleveland/`. Edit the `DATA_PATH` / `MODEL_PATH` constants at the top of the script to point at the Framingham model/data instead if needed.

### 5.5 Exploratory analysis

`eda.ipynb` contains the original exploratory data analysis and can be opened with Jupyter (`jupyter notebook eda.ipynb`) for reference; it is not part of the runtime pipeline. Or, upload the file to Google Colab.

---

## 6. Notes

**Large model files before pushing to GitHub**: `framingham/framingham_CatBoost_model.joblib` (~253 MB) exceeds GitHub's 100 MB per-file limit for a normal `git push`. Therefore:

1. If use `app.py`: Download the file from:

   - [**LINK 1**](https://drive.google.com/file/d/16s3oVKKqvprcDakxUig6t9B1zekD3aBS/view?usp=sharing) (GGDrive)
   - [**LINK 2**](https://github.com/khgthanhdinh/cvd_7/releases/download/fram_model/framingham_CatBoost_model.joblib) (Github Releases)
   - [**LINK 3**](https://huggingface.co/khgthanhdinh/fram_catboost/resolve/main/framingham_CatBoost_model.joblib?download=true) (Hugging Face)

   then **move it into** the `framingham/` folder **before** running the app.

2. If use `app2_grive.py`, `app3_git.py` or `app4_hugf.py`: they will download the file from those links, no need to move the file to the `\framingham` folder, so just run them normally:

   ```bash
   streamlit run app2_gdrive.py
   ```
