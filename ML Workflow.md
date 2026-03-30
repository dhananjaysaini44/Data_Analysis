# ML Workflow (Detailed Practical Guide for DS and AIML)

This guide is written as a practical "book + playbook" for students.

It is designed to answer not only:

- what to do

but also:

- why to do it
- when to choose a method
- what can go wrong
- how to implement it in real projects

---

## How To Use This File

For each step, you will see:

- `Goal`
- `Types`
- `Methods`
- `Advantages`
- `Disadvantages`
- `When To Use`
- `How To Implement`

This structure helps you learn concepts and make decisions in real datasets.

---

## Step 0. Problem Framing

### Goal

Convert a business question into a clear ML problem.

### Types

- `Classification`: predict category (fraud yes/no)
- `Regression`: predict number (house price)
- `Time Series Forecasting`: predict future values
- `Ranking/Recommender`: sort items by relevance
- `Clustering`: group similar entities

### Methods

- stakeholder interviews
- process mapping
- decision flow mapping
- KPI alignment

### Advantages

- prevents wrong modeling target
- avoids wasted work
- aligns technical work with business impact

### Disadvantages

- takes time upfront
- requires cross-team coordination

### When To Use

Always. No exception.

### How To Implement

Create a one-page problem statement:

1. objective
2. target definition
3. prediction horizon
4. action taken from prediction
5. success metric
6. failure cost (FP/FN cost)

---

## Step 1. Data Acquisition and Data Understanding

### Goal

Understand what data exists, where it comes from, and whether it is trustworthy.

### Types

- `Structured`: SQL tables, CSV
- `Semi-structured`: JSON logs
- `Unstructured`: text, image, audio
- `Streaming`: events, telemetry

### Methods

- schema inspection
- data profiling
- source lineage checks
- freshness checks
- uniqueness checks

### Advantages

- catches quality issues early
- reveals missing fields and wrong data types

### Disadvantages

- initial profiling can be noisy for wide tables

### When To Use

Immediately after data extraction.

### How To Implement

```python
import pandas as pd

df = pd.read_csv("data.csv")
print("shape:", df.shape)
print(df.info())
print(df.head())
print("duplicates:", df.duplicated().sum())
print("null pct:\n", (df.isnull().mean() * 100).sort_values(ascending=False).head(20))
```

Practical checks:

- primary key uniqueness
- target availability
- timestamp consistency
- impossible values

---

## Step 2. Exploratory Data Analysis (EDA)

### Goal

Understand patterns, relationships, anomalies, and business behavior in data.

### Types

- `Univariate EDA`: one variable at a time
- `Bivariate EDA`: relation between two variables
- `Multivariate EDA`: multiple variable interactions

### Methods

- summary stats
- distribution plots
- correlation heatmaps
- target-wise segmentation
- outlier plots

### Advantages

- drives feature engineering ideas
- helps detect leakage and bias

### Disadvantages

- easy to create many charts with little insight

### When To Use

Before preprocessing and before feature engineering decisions.

### How To Implement

```python
import seaborn as sns
import matplotlib.pyplot as plt

num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

print(df[num_cols].describe().T)
for c in cat_cols:
    print(c, "\n", df[c].value_counts(dropna=False).head(10))

# Example visualization
if "target" in df.columns and len(num_cols) > 0:
    sns.boxplot(data=df, x="target", y=num_cols[0])
    plt.show()
```

Practical EDA outputs should be decisions, for example:

- drop `customer_id` from features
- impute `age` by segment median
- one-hot encode `city`
- robust-scale `fare`

---

## Step 3. Data Cleaning

### Goal

Correct data quality issues that can break training and inference.

### Types

- `Structural cleaning`: duplicates, schema mismatch
- `Value cleaning`: invalid ranges, inconsistent categories
- `Format cleaning`: date parsing, unit normalization

### Methods

- deduplication
- text normalization
- data type coercion
- range clipping
- rule-based correction

### Advantages

- improves model stability
- reduces noise

### Disadvantages

- aggressive cleaning can remove useful signal

### When To Use

After EDA and before train-test split.

### How To Implement

```python
df = df.drop_duplicates()

if "gender" in df.columns:
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()

if "signup_date" in df.columns:
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

if "age" in df.columns:
    df["age"] = df["age"].clip(0, 100)
```

---

## Step 4. Missing Value Handling

### Goal

Handle missing data without introducing bias or leakage.

### Types

- `MCAR`: missing completely at random
- `MAR`: missing related to observed variables
- `MNAR`: missing not at random

### Methods

- row drop
- column drop
- mean/median/mode imputation
- constant imputation (`Unknown`, `-1`)
- group-wise imputation
- model-based imputation

### Advantages

- retains more data than dropping rows blindly
- supports robust model input preparation

### Disadvantages

- poor imputation can distort distributions

### When To Use

Always in tabular ML unless dataset is already validated complete.

### How To Implement

```python
# Group-wise median imputation example
if set(["age", "segment"]).issubset(df.columns):
    df["age"] = df.groupby("segment")["age"].transform(lambda x: x.fillna(x.median()))
```

Pipeline-safe imputation (preferred):

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
```

When to choose what:

- outliers present -> median
- stable normal data -> mean
- categorical -> mode / Unknown
- business meaning in missingness -> add missing flag

---

## Step 5. Train/Test Splitting

### Goal

Estimate true generalization performance without leakage.

### Types

- random split
- stratified split
- time-based split
- grouped split

### Methods

- `train_test_split`
- `StratifiedKFold`
- `TimeSeriesSplit`
- `GroupKFold`

### Advantages

- gives realistic evaluation
- prevents over-optimistic metrics

### Disadvantages

- wrong split strategy can silently invalidate model

### When To Use

Before fitting imputers/encoders/scalers on training data.

### How To Implement

```python
from sklearn.model_selection import train_test_split

TARGET = "target"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Use stratify for imbalanced classification.

Use time split for forecasting:

```python
df = df.sort_values("date")
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]
```

---

## Step 6. Baseline Modeling

### Goal

Set a minimal performance reference before complex models.

### Types

- dummy baseline
- simple linear baseline
- simple tree baseline

### Methods

- `DummyClassifier`, `DummyRegressor`
- `LogisticRegression`
- `LinearRegression`

### Advantages

- fast sanity check
- reveals if complex model adds real value

### Disadvantages

- may underfit strongly

### When To Use

Always before advanced model tuning.

### How To Implement

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train.select_dtypes(include="number").fillna(0), y_train)
pred = dummy.predict(X_test.select_dtypes(include="number").fillna(0))
print(classification_report(y_test, pred))
```

---

## Step 7. Encoding Categorical Features

### Goal

Convert categories into numerical representation aligned with model assumptions.

### Types

- `Label Encoding`
- `Ordinal Encoding`
- `One-Hot Encoding`
- `Frequency Encoding`
- `Target Encoding`
- `Binary/Hash Encoding`

### Methods

- sklearn preprocessors
- pandas mapping/value_counts
- external encoders (`category_encoders`)

### Advantages

- enables model training on categorical data

### Disadvantages

- wrong encoding can create fake order or leakage

### When To Use

Depends on feature type:

- nominal low cardinality -> one-hot
- ordinal -> ordinal encoding
- very high cardinality -> frequency/target/hash

### How To Implement

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

ohe = OneHotEncoder(handle_unknown="ignore")
oe = OrdinalEncoder(categories=[["low", "medium", "high"]])
```

Pros/cons summary:

- One-hot
  - advantage: no false order
  - disadvantage: dimensional explosion
- Ordinal
  - advantage: compact representation
  - disadvantage: assumes meaningful order
- Target encoding
  - advantage: strong on high cardinality
  - disadvantage: leakage risk if not CV-safe

---

## Step 8. Feature Scaling

### Goal

Normalize numeric feature magnitude for scale-sensitive models.

### Types

- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `MaxAbsScaler`
- `Normalizer`

### Methods

- sklearn scaler fit/transform
- numeric-only scaling
- pipeline-based scaling

### Advantages

- improves convergence
- stabilizes distance-based models

### Disadvantages

- unnecessary for many tree models
- can reduce interpretability of raw feature values

### When To Use

Use for:

- linear models
- SVM
- KNN
- neural networks

Usually skip for:

- tree ensembles

### How To Implement

```python
from sklearn.preprocessing import StandardScaler, RobustScaler

ss = StandardScaler()
rs = RobustScaler()
```

Selection guidance:

- normal-ish distribution -> standard scaler
- heavy outliers -> robust scaler
- bounded input expectations -> min-max scaler

---

## Step 9. Feature Engineering

### Goal

Create informative features that reflect domain behavior.

### Types

- numeric transforms (log, ratio, interactions)
- temporal features (month, weekday, recency)
- aggregations (count/mean per entity)
- binning
- text-derived features

### Methods

- domain-driven formulas
- statistical transforms
- groupby aggregations
- lag/rolling windows for time data

### Advantages

- often gives larger gains than model switching

### Disadvantages

- can introduce leakage
- can bloat feature space

### When To Use

After EDA, before final model training.

### How To Implement

```python
import numpy as np

if set(["income", "family_size"]).issubset(df.columns):
    df["income_per_member"] = df["income"] / (df["family_size"] + 1)

if "purchase_date" in df.columns:
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    df["purchase_month"] = df["purchase_date"].dt.month

if "income" in df.columns:
    df["log_income"] = np.log1p(df["income"])
```

Leakage rule:

If feature is not available at prediction time, do not use it.

---

## Step 10. Pipeline and ColumnTransformer

### Goal

Make preprocessing + modeling reproducible and deployment-safe.

### Types

- preprocessing pipeline
- full model pipeline
- custom transformer pipeline

### Methods

- `Pipeline`
- `ColumnTransformer`
- custom `BaseEstimator` + `TransformerMixin`

### Advantages

- prevents train/test mismatch
- easier serialization
- cleaner experimentation

### Disadvantages

- debugging can feel less direct initially

### When To Use

Always for production-oriented projects.

### How To Implement

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

num_cols = X_train.select_dtypes(include="number").columns
cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
```

---

## Step 11. Model Selection

### Goal

Pick candidates based on problem constraints, not hype.

### Types

- linear models
- tree-based models
- boosting models
- kernel methods
- neural networks

### Methods

- shortlist 2-4 models
- compare with same split and metric
- evaluate latency and interpretability tradeoffs

### Advantages

- practical comparison with controlled setup

### Disadvantages

- too many candidates waste time

### When To Use

After preprocessing pipeline is stable.

### How To Implement

For classification shortlist:

- logistic regression
- random forest
- gradient boosting

For regression shortlist:

- linear/ridge
- random forest regressor
- gradient boosting regressor

---

## Step 12. Evaluation Metrics

### Goal

Measure model quality according to business risk.

### Types

Classification metrics:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC

Regression metrics:

- MAE
- MSE
- RMSE
- R2

### Methods

- confusion matrix
- classification report
- residual analysis

### Advantages

- makes model performance interpretable

### Disadvantages

- wrong metric can mislead deployment decisions

### When To Use

At every training iteration.

### How To Implement

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Regression example pattern:
# mae = mean_absolute_error(y_test, y_pred_reg)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
# r2 = r2_score(y_test, y_pred_reg)
```

Metric selection by scenario:

- fraud detection: prioritize recall and PR-AUC
- medical screening: high recall + calibrated threshold
- marketing targeting: precision + uplift/business ROI

---

## Step 13. Thresholding and Calibration

### Goal

Convert probabilities into better business decisions.

### Types

- fixed threshold tuning
- cost-based thresholding
- calibrated probability models

### Methods

- precision-recall threshold sweep
- ROC/PR curve analysis
- calibration (Platt/isotonic)

### Advantages

- directly improves operational outcomes

### Disadvantages

- can overfit to one validation set if not cross-validated

### When To Use

Binary classification in production workflows.

### How To Implement

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

for t in np.arange(0.1, 0.95, 0.05):
    pred_t = (y_prob >= t).astype(int)
    print(t, precision_score(y_test, pred_t), recall_score(y_test, pred_t), f1_score(y_test, pred_t))
```

Calibration:

```python
from sklearn.calibration import CalibratedClassifierCV

cal_model = CalibratedClassifierCV(model, method="isotonic", cv=5)
```

---

## Step 14. Cross-Validation and Hyperparameter Tuning

### Goal

Get stable performance estimates and optimize model settings.

### Types

- KFold
- StratifiedKFold
- GroupKFold
- TimeSeriesSplit
- nested CV

### Methods

- `cross_val_score`
- `GridSearchCV`
- `RandomizedSearchCV`
- Bayesian optimization tools

### Advantages

- reduces split luck
- improves confidence in model selection

### Disadvantages

- computationally expensive

### When To Use

After baseline and before final model freeze.

### How To Implement

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
print(scores.mean(), scores.std())
```

---

## Step 15. Error Analysis

### Goal

Understand failure patterns and improve model where it fails.

### Types

- segment-wise errors
- false positive analysis
- false negative analysis
- temporal failure analysis

### Methods

- slice metrics by group
- inspect hardest examples
- compare feature distributions in errors vs correct predictions

### Advantages

- identifies actionable improvements

### Disadvantages

- can be time-consuming

### When To Use

After each major model iteration.

### How To Implement

```python
results = X_test.copy()
results["actual"] = y_test.values
results["pred"] = y_pred
errors = results[results["actual"] != results["pred"]]
print(errors.head())
```

---

## Step 16. Explainability and Fairness

### Goal

Make model behavior understandable and check subgroup reliability.

### Types

- global interpretability
- local interpretability
- subgroup fairness checks

### Methods

- coefficients / feature importances
- permutation importance
- SHAP
- subgroup metric reporting

### Advantages

- improves trust
- supports compliance and debugging

### Disadvantages

- explanation tools can be computationally heavy

### When To Use

Before deployment in most business-critical systems.

### How To Implement

```python
# Example subgroup metric check
tmp = X_test.copy()
tmp["actual"] = y_test.values
tmp["pred"] = y_pred

if "gender" in tmp.columns:
    for g, s in tmp.groupby("gender"):
        print(g, (s["actual"] == s["pred"]).mean())
```

---

## Step 17. Save, Version, and Reproduce

### Goal

Persist model pipeline and experiment metadata for reproducibility.

### Types

- artifact versioning
- data versioning
- experiment tracking

### Methods

- `joblib`/`pickle`
- MLflow / W&B
- DVC or data snapshot IDs

### Advantages

- reproducible results
- safe rollback

### Disadvantages

- operational overhead

### When To Use

At every model release candidate.

### How To Implement

```python
import joblib

joblib.dump(model, "model_pipeline.joblib")
model_loaded = joblib.load("model_pipeline.joblib")
```

Track metadata:

- training date
- code commit hash
- feature list
- metrics
- data snapshot ID

---

## Step 18. Inference Design

### Goal

Run predictions on new data reliably.

### Types

- batch inference
- real-time API inference
- streaming inference

### Methods

- prediction service
- scheduled scoring jobs
- queue/event-based processing

### Advantages

- operational use of ML model

### Disadvantages

- production reliability and latency constraints

### When To Use

Depends on business SLA:

- daily decisions -> batch
- instant decisions -> API/streaming

### How To Implement

```python
import pandas as pd
import joblib

model = joblib.load("model_pipeline.joblib")
new_df = pd.DataFrame([{"age": 35, "monthly_charges": 99.0, "tenure": 10, "contract_type": "month-to-month"}])
pred = model.predict(new_df)
prob = model.predict_proba(new_df)[:, 1]
print(pred[0], prob[0])
```

---

## Step 19. Deployment, Monitoring, and Drift

### Goal

Keep model healthy after go-live.

### Types

- model performance monitoring
- data drift monitoring
- operational monitoring

### Methods

- metric dashboards
- schema checks
- distribution drift checks (PSI/KS)
- alerting thresholds

### Advantages

- early issue detection
- stable business performance

### Disadvantages

- requires MLOps investment

### When To Use

Immediately after deployment.

### How To Implement

Monitor these:

- input null rates
- category frequency changes
- score distribution shifts
- business KPI degradation
- API latency/errors

---

## Step 20. Retraining and Model Lifecycle

### Goal

Refresh model when data behavior changes.

### Types

- scheduled retraining
- trigger-based retraining
- champion-challenger model lifecycle

### Methods

- drift-triggered retrain
- KPI-triggered retrain
- periodic retrain (weekly/monthly)

### Advantages

- maintains relevance

### Disadvantages

- can introduce instability if over-frequent

### When To Use

Based on drift, KPI drop, or defined cadence.

### How To Implement

Define policy:

1. retrain trigger conditions
2. validation gate metrics
3. staging test
4. rollback plan

---

## Step 21. Practical DS and AIML Concepts Map

### Data Science Essentials

- statistics and probability
- SQL
- EDA and visualization
- hypothesis testing / A-B testing

### AIML Essentials

- supervised vs unsupervised learning
- bias-variance tradeoff
- regularization
- class imbalance
- calibration
- MLOps basics

### When and How To Learn

- Start with tabular datasets first
- master one pipeline end-to-end
- then move to NLP/CV and deep learning

---

## Step 22. Real-life Mini Projects (Practice Plan)

Build these in order:

1. churn prediction (classification)
2. house price prediction (regression)
3. demand forecasting (time series)
4. sentiment or image task (NLP/CV)

For each project include:

- business framing
- EDA summary
- preprocessing choices with justification
- model comparison
- error analysis
- deployment sketch

---

## Step 23. Advantages and Disadvantages of Common Algorithms

### Logistic Regression

- Advantages: interpretable, fast, strong baseline
- Disadvantages: linear boundary limitations
- Use when: binary/multiclass tabular baseline

### Random Forest

- Advantages: handles non-linearity, robust, less preprocessing
- Disadvantages: larger models, less interpretable than linear
- Use when: tabular data with mixed feature behavior

### Gradient Boosting (XGBoost/LightGBM/CatBoost)

- Advantages: strong tabular performance
- Disadvantages: tuning complexity, overfit risk
- Use when: high-performance tabular tasks

### SVM

- Advantages: good on medium-size complex boundaries
- Disadvantages: scaling cost on large datasets
- Use when: moderate dataset and margin-based separation helps

### Neural Networks

- Advantages: flexible for complex patterns (NLP/CV/audio)
- Disadvantages: data hungry, tuning heavy, less interpretable
- Use when: large complex data and resources available

---

## Step 24. Practical Folder Structure

```text
project/
  data/
    raw/
    processed/
  notebooks/
  src/
    data/
    features/
    models/
    evaluation/
    inference/
  tests/
  artifacts/
  configs/
  README.md
  requirements.txt
```

Why this helps:

- reproducibility
- collaboration
- easier deployment transition

---

## Step 25. Common Mistakes and Fixes

### Mistake: preprocessing before split

- Problem: leakage
- Fix: split first, fit transforms on train only

### Mistake: wrong metric

- Problem: misleading results
- Fix: align metric with business cost

### Mistake: no baseline

- Problem: no reference point
- Fix: start with dummy/simple model

### Mistake: no monitoring

- Problem: model silently degrades
- Fix: production KPI + drift monitoring

---

## Step 26. Production Readiness Checklist

Before deployment, confirm:

- problem and KPI are clear
- leakage checks passed
- split strategy valid
- preprocessing pipeline reproducible
- model metrics acceptable for business
- threshold tuned (if classification)
- artifact versioned
- inference tested on fresh samples
- monitoring and alerting configured
- rollback model ready

---

## Step 27. 90-Day Learning Plan

### Month 1

- Python + pandas + SQL
- EDA + cleaning + missing values

### Month 2

- encoding + scaling + pipelines
- baseline + model comparison + metrics

### Month 3

- tuning + error analysis + explainability
- deployment sketch + monitoring basics
- portfolio projects

---

## Step 28. Final Guidance

Think like a data professional:

1. understand business
2. understand data
3. build leakage-safe baseline
4. evaluate correctly
5. improve with evidence
6. deploy with monitoring

Do not optimize for "most complex model."

Optimize for:

- correctness
- reproducibility
- usefulness
- reliability

That is real DS and AIML engineering.

---

## Step-by-Step Deep Dive (Types, Method Choice, Pros/Cons, Common Errors)

This section expands each step with decision-grade detail.

Use this section while implementing a real project.

---

### Step 0 Deep Dive: Problem Framing

#### Types and when to use

- Classification
  - Use when target is class label (`yes/no`, `spam/not spam`).
  - Prefer if downstream action is categorical.
- Regression
  - Use when target is continuous numeric value.
  - Prefer when business needs a number (price, demand).
- Forecasting
  - Use when time is central and target is future value.
  - Prefer when seasonality/trend matter.

#### Methods and preference

- Stakeholder interviews
  - Prefer in business-facing projects to clarify decision flow.
- KPI-to-metric mapping
  - Prefer when multiple metrics conflict.
- Cost matrix definition
  - Prefer for high-stakes classification (fraud, medical).

#### Pros and cons

- Strong framing
  - Pros: reduces rework, aligns model with impact.
  - Cons: slower start.
- Weak framing
  - Pros: quick to start coding.
  - Cons: high risk of solving wrong problem.

#### Common errors

- ambiguous target definition
- using proxy target with future leakage
- metric not aligned with business cost

---

### Step 1 Deep Dive: Data Acquisition and Understanding

#### Types

- Batch tables (warehouse, CSV exports)
- Event logs (JSON, streaming)
- External APIs (third-party data)

#### Methods and preference

- Schema profiling
  - Prefer first for all sources.
- Data lineage check
  - Prefer for regulated domains.
- Freshness/latency check
  - Prefer if inference is near real-time.

#### Pros and cons

- Deep profiling
  - Pros: catches data issues early.
  - Cons: effort on very wide tables.

#### Common errors

- mixed data types in same column
- timezone inconsistencies
- silently truncated exports
- duplicate rows due to bad joins

---

### Step 2 Deep Dive: EDA

#### Types

- Univariate: distribution, skew, outliers.
- Bivariate: feature-target relationship.
- Multivariate: interactions and confounding.

#### Methods and preference

- Histogram/KDE
  - Prefer for numeric distribution shape.
- Boxplot
  - Prefer for outlier detection.
- Crosstab/target rate by category
  - Prefer for categorical predictive signal.
- Correlation heatmap
  - Prefer for quick multicollinearity scan (numeric only).

#### Pros and cons

- Rich EDA
  - Pros: better feature strategy.
  - Cons: analysis paralysis if not decision-focused.

#### Common errors

- overinterpreting correlation as causation
- ignoring sample size in subgroup insights
- plotting without checking data quality first

---

### Step 3 Deep Dive: Data Cleaning

#### Types

- Structural cleaning (`duplicates`, schema fixes)
- Value cleaning (invalid ranges, typos)
- Semantic cleaning (business-rule consistency)

#### Methods and preference

- Deduplication
  - Prefer when same entity-event appears multiple times.
- Category normalization
  - Prefer for messy free-text categories.
- Rule-based clipping
  - Prefer for known valid ranges.

#### Pros and cons

- Aggressive cleaning
  - Pros: cleaner model input.
  - Cons: may remove rare but valid cases.

#### Common errors

- deleting rows without reason log
- normalizing categories inconsistently across train/test
- converting dates with wrong locale

---

### Step 4 Deep Dive: Missing Values

#### Types

- MCAR: random missingness
- MAR: missingness linked to observed fields
- MNAR: missingness tied to unobserved process

#### Methods and preference

- Drop rows
  - Prefer when missing fraction is tiny and unbiased.
- Drop columns
  - Prefer when column has very high missingness and low value.
- Median imputation
  - Prefer for skewed numeric/outliers.
- Mean imputation
  - Prefer for near-normal numeric.
- Most frequent
  - Prefer for categorical.
- Constant (`Unknown`)
  - Prefer when missingness itself is meaningful.
- Group-wise imputation
  - Prefer when subgroup behavior differs strongly.

#### Pros and cons

- Simple imputation
  - Pros: fast, stable.
  - Cons: can shrink variance.
- Model-based imputation
  - Pros: potentially better accuracy.
  - Cons: complexity and leakage risk.

#### Common errors

- imputing before split (leakage)
- fitting imputer on full data
- using mean for heavily skewed data

---

### Step 5 Deep Dive: Splitting Strategy

#### Types

- Random split
- Stratified split
- Group split
- Time-based split

#### Methods and preference

- `train_test_split(..., stratify=y)`
  - Prefer for imbalanced classification.
- `GroupKFold`
  - Prefer when same entity appears many times.
- `TimeSeriesSplit`
  - Prefer for time-dependent data.

#### Pros and cons

- Random split
  - Pros: simple.
  - Cons: invalid for grouped/time data.
- Time split
  - Pros: realistic.
  - Cons: lower effective training signal in early windows.

#### Common errors

- random split on time series
- same customer in train and test
- no fixed random seed (non-reproducible results)

---

### Step 6 Deep Dive: Baseline Modeling

#### Types

- Dummy baseline
- Simple linear baseline
- Simple tree baseline

#### Methods and preference

- Dummy classifier/regressor
  - Prefer to set absolute floor.
- Logistic/Linear
  - Prefer as interpretable baseline.

#### Pros and cons

- Baseline-first
  - Pros: guards against fake progress.
  - Cons: may feel "too simple".

#### Common errors

- skipping baseline and overfitting complex model
- comparing models with different preprocessing

---

### Step 7 Deep Dive: Encoding

#### Types and when to prefer

- One-hot
  - Prefer nominal low-cardinality categories.
- Ordinal
  - Prefer true ordered categories.
- Label encoding
  - Prefer target labels or tree-only quick prototypes.
- Frequency encoding
  - Prefer high-cardinality nominal with limited memory.
- Target encoding
  - Prefer high-cardinality with strong signal and CV-safe setup.

#### Pros and cons

- One-hot
  - Pros: no fake order.
  - Cons: sparse/high-dimensional.
- Ordinal
  - Pros: compact.
  - Cons: incorrect if order is fake.
- Target encoding
  - Pros: powerful.
  - Cons: high leakage risk.

#### Common errors

- using ordinal on nominal variables
- not handling unknown categories at inference
- target encoding without out-of-fold strategy

---

### Step 8 Deep Dive: Scaling

#### Types and preference

- StandardScaler
  - Prefer default for linear/SVM/KNN.
- MinMaxScaler
  - Prefer bounded range needs (0-1).
- RobustScaler
  - Prefer heavy outliers.

#### Pros and cons

- StandardScaler
  - Pros: stable default.
  - Cons: sensitive to outliers.
- RobustScaler
  - Pros: outlier-resistant.
  - Cons: less intuitive transformed values.

#### Common errors

- scaling tree-only models unnecessarily
- scaling one-hot dummies without reason
- fitting scaler on full dataset

---

### Step 9 Deep Dive: Feature Engineering

#### Types

- Ratio features
- Interaction features
- Temporal features
- Aggregation features
- Binning
- Log transforms

#### Methods and preference

- Ratios
  - Prefer when denominator has business meaning.
- Log transform
  - Prefer highly skewed positive variables.
- Aggregations
  - Prefer transactional/entity datasets.
- Temporal decomposition
  - Prefer date-driven behavior problems.

#### Pros and cons

- Domain features
  - Pros: strong lift potential.
  - Cons: maintenance overhead.

#### Common errors

- engineered features using future info
- division by zero in ratios
- creating unstable features from sparse groups

---

### Step 10 Deep Dive: Pipeline/ColumnTransformer

#### Types

- Preprocessing-only pipeline
- Full train/infer pipeline
- Custom transformer pipeline

#### Methods and preference

- `Pipeline + ColumnTransformer`
  - Prefer default production pattern.
- Custom transformer
  - Prefer for reusable business logic.

#### Pros and cons

- Unified pipeline
  - Pros: reproducible, leakage-safe.
  - Cons: debugging learning curve.

#### Common errors

- preprocessing done outside pipeline
- forgetting `handle_unknown="ignore"`
- mismatched columns at inference

---

### Step 11 Deep Dive: Model Selection

#### Types

- Interpretable models
- Ensemble models
- Margin-based models
- Deep models

#### Methods and preference

- Logistic/Linear
  - Prefer interpretability and baseline speed.
- RandomForest
  - Prefer strong non-linear baseline.
- Gradient boosting
  - Prefer high-performance tabular tasks.
- Deep learning
  - Prefer large complex data (text/image/audio).

#### Pros and cons

- Complex models
  - Pros: better fit potential.
  - Cons: harder to explain and tune.

#### Common errors

- selecting model before understanding data constraints
- ignoring latency and inference cost

---

### Step 12 Deep Dive: Evaluation

#### Types

- Classification metrics
- Regression metrics
- Ranking/retrieval metrics

#### Methods and preference

- F1/PR-AUC
  - Prefer imbalanced classification.
- Recall
  - Prefer when false negatives are costly.
- MAE
  - Prefer business-friendly regression errors.
- RMSE
  - Prefer when large errors are extra costly.

#### Pros and cons

- Single metric optimization
  - Pros: simpler.
  - Cons: can hide failure modes.

#### Common errors

- using accuracy for rare-event problems
- comparing models on different splits
- evaluating thresholded outputs only (ignore probability quality)

---

### Step 13 Deep Dive: Thresholding and Calibration

#### Types

- Threshold optimization
- Cost-sensitive thresholding
- Probability calibration

#### Methods and preference

- PR-curve threshold sweep
  - Prefer imbalanced classification.
- Cost function optimization
  - Prefer when FP/FN costs known.
- Isotonic/Platt calibration
  - Prefer when probability decisions are used.

#### Pros and cons

- Calibration
  - Pros: better risk scores.
  - Cons: extra CV complexity.

#### Common errors

- fixed 0.5 threshold by habit
- calibrating on same data used for final test

---

### Step 14 Deep Dive: CV and Tuning

#### Types

- Grid search
- Random search
- Bayesian search
- Nested CV

#### Methods and preference

- RandomizedSearchCV
  - Prefer practical speed-performance balance.
- GridSearchCV
  - Prefer small search spaces.
- Nested CV
  - Prefer rigorous research-grade selection.

#### Pros and cons

- Broad tuning
  - Pros: better model potential.
  - Cons: compute/time cost.

#### Common errors

- tuning without baseline
- data leakage inside CV folds
- searching too many parameters blindly

---

### Step 15 Deep Dive: Error Analysis

#### Types

- class-wise errors
- segment-wise errors
- time-window errors
- confidence-based errors

#### Methods and preference

- confusion matrix slices
  - Prefer classification debugging.
- residual diagnostics
  - Prefer regression debugging.
- subgroup breakdown
  - Prefer fairness and robustness checks.

#### Pros and cons

- detailed analysis
  - Pros: targeted improvements.
  - Cons: additional analysis time.

#### Common errors

- optimizing globally but failing on key segments
- ignoring hard examples near decision boundary

---

### Step 16 Deep Dive: Explainability and Fairness

#### Types

- Global explanation
- Local explanation
- Fairness subgroup analysis

#### Methods and preference

- Coefficients
  - Prefer linear models.
- Feature importance
  - Prefer tree ensembles (quick view).
- SHAP
  - Prefer local + global explainability.
- Subgroup metrics
  - Prefer fairness/compliance checks.

#### Pros and cons

- Explainability tooling
  - Pros: trust and debugging.
  - Cons: compute overhead, potential misinterpretation.

#### Common errors

- treating importance as causality
- fairness checks only on overall accuracy

---

### Step 17 Deep Dive: Versioning and Reproducibility

#### Types

- artifact versioning
- data versioning
- experiment metadata logging

#### Methods and preference

- `joblib` for artifacts
- MLflow/W&B for experiment tracking
- DVC/data snapshots for training data lineage

#### Pros and cons

- full lineage
  - Pros: reliable rollback, auditability.
  - Cons: setup overhead.

#### Common errors

- saving model without preprocessing pipeline
- no data snapshot reference
- no seed or env dependency lock

---

### Step 18 Deep Dive: Inference

#### Types

- batch scoring
- synchronous API scoring
- asynchronous streaming scoring

#### Methods and preference

- Batch
  - Prefer periodic decisions.
- API
  - Prefer low-latency decisions.
- Streaming
  - Prefer event-time continuous processing.

#### Pros and cons

- API inference
  - Pros: immediate action.
  - Cons: reliability/latency engineering needs.

#### Common errors

- schema mismatch between train and inference payload
- missing category handling failures
- no timeout/retry behavior in service

---

### Step 19 Deep Dive: Monitoring and Drift

#### Types

- data drift
- concept drift
- label drift
- operational drift (latency/errors)

#### Methods and preference

- feature distribution monitoring
- score distribution monitoring
- business KPI monitoring
- alert thresholds + incident playbooks

#### Pros and cons

- detailed monitoring
  - Pros: early failure detection.
  - Cons: alert fatigue if thresholds poor.

#### Common errors

- monitoring only model metric, not input drift
- no baseline reference window
- no defined action after alert

---

### Step 20 Deep Dive: Retraining and Lifecycle

#### Types

- periodic retraining
- trigger-based retraining
- champion-challenger rollout

#### Methods and preference

- fixed schedule
  - Prefer stable seasonal data.
- drift-triggered
  - Prefer dynamic environments.
- champion-challenger
  - Prefer safer incremental upgrades.

#### Pros and cons

- frequent retraining
  - Pros: adapts quickly.
  - Cons: potential instability and ops load.

#### Common errors

- retraining without validation gates
- replacing production model without rollback plan
- forgetting to compare against incumbent champion

---

### Step 21 Deep Dive: DS/AIML Concept Map

#### Practical method preference

- Start with tabular supervised learning first.
- Learn SQL and statistics alongside ML.
- Add deep learning after strong tabular fundamentals.

#### Common errors

- learning advanced models before mastering data fundamentals

---

### Step 22 Deep Dive: Mini Project Strategy

#### Method preference

- Start with one end-to-end project per problem family.
- Use identical documentation template across projects.

#### Common errors

- many half-finished notebooks
- no evaluation or deployment narrative

---

### Step 23 Deep Dive: Algorithm Tradeoff Choice

#### Preference guide

- Need interpretability -> linear models first
- Need strong tabular performance -> boosting
- Need robustness + low tuning -> random forest
- Need unstructured data power -> deep learning

#### Common errors

- choosing model by popularity, not constraints

---

### Step 24 Deep Dive: Project Structure

#### Method preference

- Keep feature logic in `src/features`.
- Keep evaluation code in `src/evaluation`.
- keep tests for preprocessing and inference.

#### Common errors

- mixing notebook experiments and production code

---

### Step 25 Deep Dive: Failure Prevention

#### Method preference

- maintain a pre-deployment checklist and "stop ship" criteria
- log all data cleaning and feature assumptions

#### Common errors

- undocumented assumptions
- silent changes in upstream data

---

### Step 26 Deep Dive: Production Checklist Use

#### Method preference

- convert checklist to CI gate where possible
- require sign-off for metric regressions

#### Common errors

- checklist exists but not enforced in workflow

---

### Step 27 Deep Dive: 90-Day Plan Execution

#### Method preference

- Weekly cadence:
  - 3 days build
  - 1 day review
  - 1 day documentation

#### Common errors

- learning without project output

---

### Step 28 Deep Dive: Professional Mindset

#### Method preference

- choose reliability over complexity
- choose reproducibility over fast hacks
- choose clear metrics over vague claims

#### Common errors

- optimizing leaderboard number without business validity

