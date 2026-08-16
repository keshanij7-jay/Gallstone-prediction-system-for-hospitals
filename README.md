#  Gallstone Disease Risk Prediction-system-for-hospitals

### A Non-Imaging Machine Learning Screening Model

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4B8BBE?style=flat)
![Pandas](https://img.shields.io/badge/pandas-2.0-150458?style=flat&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

> **Predict gallstone disease from routine blood tests and body composition —
> no ultrasound required.**  
> Built with logistic regression, random forest, and full statsmodels
> statistical validation on 319 patients from Ankara VM Medical Park Hospital.

---

##  Results at a Glance

| | Logistic Regression | Random Forest |
|---|:---:|:---:|
| **Test AUC** | **0.884**  | 0.865 |
| **Test Accuracy** | **78.1%** | 78.1% |
| **CV AUC** (5-fold) | 0.829 | 0.842 |
| **CV Accuracy** (5-fold) | 0.737 | 0.749 |
| **McFadden R²** | **0.521**  | — |
| **AIC / BIC** | **247.2 / 385.3** | — |

** Logistic Regression selected as final model**  higher test AUC + full
clinical interpretability.

---

##  Why This Matters

Gallstone disease affects **~20% of adults worldwide.** Standard diagnosis =
abdominal ultrasound — expensive, slow, not always available.

```
The gap:  Clinicians need risk stratification BEFORE ordering imaging.
The fix:  A risk score from data the clinic already has.
```

An AUC of **0.884** means the model correctly ranks a high-risk patient above
a low-risk patient **88.4% of the time** — enough to meaningfully reduce
unnecessary scans.

---

##  Dataset

| Property | Value |
|---|---|
| Source | Ankara VM Medical Park Hospital, Turkey |
| Period | June 2022 – June 2023 |
| Patients | **319 anonymised records** |
| Ethics | E2-23-4632 |
| Features | **38** (demographic + bioimpedance + laboratory) |
| Missing values | **0** — complete dataset |
| Class balance | 50.5% negative / 49.5% positive |
| Train / Test | 255 / 64 (80/20 stratified) |

---

##  Pipeline

```
Raw CSV
  │
  ├─ Column normalisation  (snake_case, regex cleaning)
  ├─ Binary target encode  (Yes/No → 1/0)
  ├─ EDA                   (boxplots, heatmap, class balance)
  ├─ Stratified split      (80/20, random_state=42)
  │
  ├─ Logistic Regression Pipeline
  │    ├─ StandardScaler   (numeric features)
  │    ├─ OneHotEncoder    (categorical features, drop='first')
  │    └─ LogisticRegression(max_iter=1000, class_weight='balanced')
  │
  ├─ Random Forest Pipeline
  │    ├─ StandardScaler
  │    ├─ OneHotEncoder
  │    └─ RandomForestClassifier(n_estimators=500, class_weight='balanced')
  │
  ├─ 5-Fold Stratified CV
  ├─ Test set evaluation
  └─ statsmodels refit     (p-values, CI, McFadden R², AIC/BIC)
```

> **Key design:** Each pipeline gets its own independent `ColumnTransformer`
> instance — preventing shared-state fitting conflicts and `NotFittedError`.

---

##  Model Evaluation

### Confusion Matrix — Logistic Regression (n = 64)

```
                   Predicted No    Predicted Yes
  Actual No             26               6        (FP = 6)
  Actual Yes             8              24        (FN = 8)
```

| Class | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| 0 — No gallstones | 0.765 | 0.812 | 0.788 |
| 1 — Gallstones | 0.800 | 0.750 | 0.774 |
| **Macro avg** | **0.782** | **0.781** | **0.781** |

> In a screening context, **false negatives (FN = 8)** are the critical
> error — a missed gallstone patient risks delayed treatment. The model
> keeps FN low while maintaining balanced precision and recall.

### ROC-AUC Scale

```
0.50  ████░░░░░░░░░░░░░░░░  Random guessing
0.70  ██████████░░░░░░░░░░  Acceptable
0.88  ████████████████░░░░  ← This model (Excellent)
1.00  ████████████████████  Perfect
```

---

##  Statistical Validation

Re-fitted with `statsmodels.Logit` for formal inference.

### Goodness-of-Fit

| Metric | Value | Benchmark |
|---|:---:|---|
| **McFadden pseudo R²** | **0.521** | > 0.40 = outstanding (rare in medicine) |
| **Log-Likelihood** | -84.600 | Full model |
| **LL-Null** | -176.730 | Intercept-only |
| **LLR p-value** | **4.29 × 10⁻²¹** | Highly significant vs null |
| **AIC** | **247.2** | Lower = better fit/complexity balance |
| **BIC** | **385.3** | Stricter complexity penalty |
| **Observations** | 255 | Training set |

> McFadden R² of **0.521** is exceptional — published clinical models
> typically achieve 0.20–0.35. This confirms the predictor set captures
> genuine biological signal.

### Logistic Regression Equation

```
logit(P) = β₀ + β₁·Age + β₂·Gender + β₃·CRP + β₄·HGB
         + β₅·VitaminD + β₆·AST + β₇·DM + β₈·CAD + ...

P(gallstone) = 1 / (1 + e^−logit(P))
```

---

##  Significant Predictors (p < 0.05)

| Feature | Coef (β) | z | p-value | 95% CI | Direction |
|---|:---:|:---:|:---:|---|:---:|
| **Vitamin D** | -1.166 | -4.265 | **0.0000** | [-1.702, -0.630] | 🔽 Protective |
| **Gender** | -3.578 | -2.855 | **0.0043** | [-6.033, -1.122] | 🔽 Male = lower risk |
| **CRP** | +1.995 | +2.757 | **0.0058** | [+0.577, +3.413] | 🔺 Raises risk |
| **Haemoglobin** | -0.906 | -2.473 | **0.0134** | [-1.624, -0.188] |  Protective |
| **Age** | +1.383 | +2.396 | **0.0166** | [+0.252, +2.514] |  **Raises risk** |
| **CAD** | -0.601 | -2.220 | **0.0264** | [-1.131, -0.070] |  Lowers risk |
| **AST** | -1.178 | -2.175 | **0.0296** | [-2.240, -0.117] |  Lowers risk |
| **Diabetes (DM)** | +0.674 | +2.164 | **0.0305** | [+0.064, +1.285] |  **Raises risk** |

All confidence intervals exclude zero → statistically significant at α = 0.05.

### Random Forest — Top Feature Importances

| Rank | Feature | Importance |
|---|---|:---:|
| 1 | c_reactive_protein_crp | 0.152 |
| 2 | vitamin_d | 0.092 |
| 3 | extracellular_fluid_total_body_water_ecf_tbw | 0.045 |
| 4 | aspartat_aminotransferaz_ast | 0.038 |

> **CRP** and **Vitamin D** are top predictors in **both** models —
> linear (significant p-values) and non-linear (highest importance scores).
> Cross-model agreement strengthens confidence in these findings.

---

##  Limitations

| | |
|---|---|
|  Single centre | One hospital in Turkey — external validity unverified |
|  Retrospective | Historical data only — prospective validation needed before deployment |
|  Sample size | n=319 sufficient for pilot, small for clinical deployment |
|  Missing features | No diet, activity, genetic, or family history data |
|  Multicollinearity | LDL ↔ Total Cholesterol (r ≈ 0.82) — some coefficients less stable |
|  Convergence | `statsmodels` reports non-convergence — driven by quasi-complete separation in hyperlipidaemia (extremely wide CI, should not be independently interpreted) |

---

##  Stage 2 — Roadmap

Stage 1 answers: *does this patient have gallstones?*  
Stage 2 answers three deeper questions for confirmed positive patients:

```
Stage 1 positive patient
        │
        ├── Model A → Stone size (mm) + Severity: Mild / Moderate / Severe
        │              Method: Linear regression + Ordinal classification
        │
        ├── Model B → Complication risk within 12 months
        │              (cholecystitis / pancreatitis / biliary obstruction)
        │              Method: Logistic Regression + Gradient Boosting
        │
        └── Model C → Post-treatment recurrence probability + time to recurrence
                       Method: Logistic Regression + Cox Proportional Hazards
```

> Stage 2 requires longitudinal follow-up data — design proposal available
> in [`Stage2_Proposal.docx`](Stage2_Proposal.docx).

---

##  Tech Stack

| Library | Version | Use |
|---|---|---|
| `Python` | 3.11 | Core language |
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `scikit-learn` | ≥ 1.3 | ML pipeline, models, metrics |
| `statsmodels` | ≥ 0.14 | p-values, CI, McFadden R², AIC/BIC |
| `matplotlib` | ≥ 3.7 | ROC curves, coefficient plots |
| `seaborn` | ≥ 0.12 | Heatmap, boxplots |

---

##  How to Run

```bash
# 1. Clone
git clone https://github.com/yourusername/gallstone-prediction.git
cd gallstone-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add dataset
# Place gallstone_dataset.csv in the project root

# 5. Launch notebook
jupyter notebook stage_1.ipynb
# Then: Kernel → Restart & Run All
```

---

##  Project Structure

```
gallstone-prediction/
├── stage_1.ipynb            # Full pipeline notebook
├── REPORT.md                # Written project report
├── Stage2_Proposal.docx     # Stage 2 design proposal
├── README.md                # This file
├── requirements.txt         # Dependencies
└── gallstone_dataset.csv    # Dataset (place here before running)
```

---

##  Quick Stats Card

```
┌─────────────────────────────────────────────────┐
│         GALLSTONE PREDICTION — STAGE 1          │
├─────────────────────────────────────────────────┤
│  Dataset     319 patients · 38 features · 0 NaN │
│  Split       255 train / 64 test (80/20)         │
│  CV          5-fold stratified                   │
├─────────────────────────────────────────────────┤
│  AUC         0.884  (Excellent)                  │
│  Accuracy    78.1%                               │
│  F1-score    0.781  (macro avg)                  │
├─────────────────────────────────────────────────┤
│  McFadden R² 0.521  (Outstanding)               │
│  AIC / BIC   247.2 / 385.3                      │
│  LLR p-value 4.29 × 10⁻²¹                      │
├─────────────────────────────────────────────────┤
│  Top predictors (p < 0.05)                       │
│  Vitamin D · CRP · Age · Gender · Diabetes      │
└─────────────────────────────────────────────────┘
```

---

*Dataset ethically approved (E2-23-4632) · All records anonymised*  
*Research prototype — not validated for clinical deployment*

