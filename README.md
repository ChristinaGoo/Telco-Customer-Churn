# Telco Customer Churn

Predict which telecom customers are likely to churn and rank them by risk so the business can prioritise retention actions (offers, outreach, product changes) for high-risk customers.

---

## Problem statement

**Goal:** Predict which customers are likely to churn (binary label or probability per customer).

**Use of predictions:** Prioritise retention actions—e.g. target the top 20% at-risk customers for campaigns, discounts, or support.

**Success:** A model that ranks customers by churn risk and supports segmenting by risk tier (High / Medium / Low) for actionable retention.

---

## Tools used

| Category        | Tools |
|----------------|------|
| **Language**   | Python 3 |
| **Editor / IDE** | [VS Code](https://code.visualstudio.com/) |
| **Data**       | [pandas](https://pandas.pydata.org/) |
| **ML**         | [scikit-learn](https://scikit-learn.org/) (Logistic Regression, Random Forest, metrics, preprocessing) |
| **Visualisation** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/python/) |
| **Notebook**   | Jupyter (ipykernel) |
| **Persistence**| [joblib](https://joblib.readthedocs.io/) (saved model and scaler) |

---

## Workflow

The analysis is implemented in a single Jupyter notebook: `notebooks/churn.ipynb`.

1. **EDA** — Load raw data, inspect shape and types, target balance, numeric and categorical distributions, correlations, churn rates by key variables.
2. **Data preprocessing** — Impute missing `TotalCharges`, drop identifier, encode target (Churn 1/0), binary/ordinal/one-hot encoding for categoricals; verify numeric-only, no missing values.
3. **Problem statement** — Define goal (churn prediction and ranking) and use of predictions (retention campaigns).
4. **Modeling** — Train/validation split (stratified 75/25), scale features (`StandardScaler`, fit on train); baseline (stratified dummy); train **Logistic Regression** and **Random Forest** (with `class_weight='balanced'`); compare with **Average Precision** and **ROC-AUC**; select best model.
5. **Evaluation** — Confusion matrix on validation set; feature importance (coefficients for LR, importances for RF).
6. **Churn probability and risk tier** — Rank customers by predicted churn probability; assign risk tiers (High / Medium / Low); report top 20% at-risk and churn rate in that segment.
7. **Save artifact** — Retrain best model on full data (train + val) and save to `models/churn_model.joblib` (model, scaler, feature names, target name).
8. **Results summary** — Tables and short narrative on data, split, baseline, models, best model, actionable output, artifact, and top drivers with recommended actions.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Telco_Customer_Churn.git
cd Telco_Customer_Churn
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux (on Windows use: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 3. Add the data

Place the Telco Customer Churn dataset in the raw data folder:

- **Path:** `data/raw/Telco-Customer-Churn.csv`
- **Source:** e.g. [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) or equivalent (21 columns: customerID, demographics, tenure, services, contract, billing, Churn).

### 4. Run the notebook

From the project root:

```bash
jupyter notebook notebooks/churn.ipynb
```

Or open `notebooks/churn.ipynb` in Jupyter Lab / VS Code and run all cells. The notebook expects to be run from the `notebooks/` directory (paths use `../data/` and `../models/`).

---

## Project structure

```
Telco_Customer_Churn/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Telco-Customer-Churn.csv
│   ├── interim/
│   ├── processed/
│   └── external/
├── models/                     # churn_model.joblib (after running the notebook)
├── notebooks/
│   └── churn.ipynb            # Full pipeline: EDA → preprocessing → modeling → risk tier → save
├── references/
├── reports/
└── src/
```

---

## Output

- **Model artifact:** `models/churn_model.joblib` — trained model, fitted scaler, `feature_cols`, `target_col` for scoring new customers.
- **Notebook:** Ranked validation customers by churn probability, risk tiers (High / Medium / Low), and top drivers with suggested retention actions.
