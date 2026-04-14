# Data Analysis

This repository is a notebook-first learning workspace for Python-based data analysis and ML practice. It includes foundational exercises, preprocessing methods, encoding/scaling practice, classification/regression workflows, and a detailed end-to-end ML handbook.

## What This Project Covers

- NumPy and pandas fundamentals
- Data cleaning and missing-value handling
- EDA and visualization with matplotlib/seaborn
- Categorical encoding and feature scaling
- Titanic and house-price style modeling practice
- End-to-end ML engineering workflow guidance

## Project Files

### Notebooks

- `FSML_1.ipynb` - NumPy foundations and introductory ML setup
- `FSML_2.ipynb` - pandas basics with `Fortune_10.csv`
- `FSML_3.ipynb` - grouping/aggregation with student datasets
- `FSML_5_&_6.ipynb` - cleaning + visualization workflows on `train.csv`
- `P_1_FSML_5_&_6.ipynb` - preprocessing strategy (columns-first)
- `FSML_7.ipynb` - preprocessing strategy (rows-first)
- `FSML_8.ipynb` - extended missing-value strategy on `train.csv`
- `FSML_9.ipynb` - encoding/scaling experiments across multiple datasets
- `P_2_FSML_9.ipynb` - extended Titanic-style preprocessing/modeling flow
- `FSML_10.ipynb` - advanced continuation notebook in the FSML sequence
- `Credit_Risk.ipynb` - credit-risk classification practice
- `KNN.ipynb` - K-nearest neighbors practice notebook
- `Polynomial Regression.ipynb` - polynomial regression walkthrough
- `hapur_house_price_prediction_model.ipynb` - house-price prediction workflow
- `hapur_house_price_prediction_model_corrected.ipynb` - refined house-price workflow

### Documents

- `ML_Workflow.md` - comprehensive practical ML workflow guide (from problem framing to deployment habits)
- `environment.yml` - conda environment specification for this project

### Datasets (included in repository)

- `Bengaluru_House_Data.csv`
- `Fortune_10.csv`
- `Who_is_responsible_for_global_warming.csv`
- `credit_risk_dataset.csv`
- `train.csv`
- `tips.csv`
- `fruits.csv`
- `students_dataset.txt`
- `students_dirty_data.csv`
- `df1_students.txt`
- `df2_marks.txt`
- `ghaziabad_house_price_dataset_1000.csv`
- `hapur_house_price_dataset.csv`
- `titanic_AfterEDA.csv`
- `titanic after encoding.csv`

### Generated Visual Outputs

Notebook-generated chart exports such as `.png`, `.svg`, `.pdf`, and `.html` files are treated as derived artifacts and should stay out of Git unless they are intentionally curated for documentation.

## Tech Stack

- Python 3.12
- Jupyter Notebook
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Setup

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate data-analysis
jupyter notebook
```

### Option B: venv + pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn notebook ipykernel
jupyter notebook
```

## Suggested Learning Flow

1. `FSML_1.ipynb`
2. `FSML_2.ipynb`
3. `FSML_3.ipynb`
4. `FSML_5_&_6.ipynb`
5. `P_1_FSML_5_&_6.ipynb`
6. `FSML_7.ipynb`
7. `FSML_8.ipynb`
8. `FSML_9.ipynb`
9. `P_2_FSML_9.ipynb`
10. `FSML_10.ipynb`
11. `Credit_Risk.ipynb`
12. `KNN.ipynb`
13. `Polynomial Regression.ipynb`
14. `hapur_house_price_prediction_model.ipynb`
15. `hapur_house_price_prediction_model_corrected.ipynb`

## Git Hygiene

- Commit notebooks, markdown guides, environment files, and the small teaching datasets used by the exercises.
- Ignore local-only artifacts such as `.idea/`, `.ipynb_checkpoints/`, virtual environments, caches, logs, and exported plots.
- If you later reorganize the repository, prefer `data/` for raw datasets and `outputs/` for derived charts so Git tracking rules stay simple.

## Positive Improvement Opportunities

- Move datasets/images into dedicated folders like `data/` and `outputs/` for cleaner navigation.
- Add a short objective + expected output section at the top of each notebook for faster revision.
- Add reusable preprocessing utilities in a small `src/` package to reduce repeated notebook code.
- Add one benchmark metrics table in the README to track progress across classification/regression notebooks.
