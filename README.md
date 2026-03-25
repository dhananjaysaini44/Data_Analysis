# Data Analysis

This repository contains a set of Jupyter notebooks for practicing data analysis, data cleaning, visualization, and categorical encoding with Python. The work is organized as notebook-based exercises using small CSV/TXT datasets and a larger `train.csv` dataset for preprocessing and exploratory analysis.

## Project Focus

- NumPy basics and array operations
- Pandas data loading, inspection, and transformation
- Grouping and summarizing tabular data
- Data cleaning and missing-value handling
- Exploratory data analysis with Seaborn and Matplotlib
- Feature encoding with `pandas` and `scikit-learn`

## Repository Structure

### Notebooks

- `FSML_1.ipynb`  
  Introductory practice with `numpy`, basic arrays, shapes, dtypes, and simple `scikit-learn` usage.

- `FSML_2.ipynb`  
  Pandas fundamentals using `Fortune_10.csv`, including dataset inspection and basic cleaning operations.

- `FSML_3.ipynb`  
  Grouping, aggregation, and analysis with student datasets such as `students_dataset.txt`, `df1_students.txt`, `df2_marks.txt`, and `students_dirty_data.csv`.

- `FSML_5_&_6.ipynb`  
  Data visualization and preprocessing work using `Who_is_responsible_for_global_warming.csv` and `train.csv`, including heatmaps, histograms, and KDE plots.

- `P_1_FSML_5_&_6.ipynb`  
  Preprocessing workflow labeled as "Method 1 - Columns first then rows" on `train.csv`.

- `FSML_7.ipynb`  
  Alternative preprocessing workflow labeled as "Method 2 - Rows first then Columns" on `train.csv`.

- `FSML_8.ipynb`  
  Extended version of Method 2 with class-based mean imputation for missing values in `train.csv`.

- `FSML_9.ipynb`  
  Categorical encoding techniques using `tips.csv` and `fruits.csv`, including `get_dummies`, `LabelEncoder`, `OrdinalEncoder`, and `OneHotEncoder`.

### Datasets

The repository includes several sample datasets used directly by the notebooks:

- `Fortune_10.csv`
- `Who_is_responsible_for_global_warming.csv`
- `train.csv`
- `tips.csv`
- `fruits.csv`
- `students_dataset.txt`
- `students_dirty_data.csv`
- `df1_students.txt`
- `df2_marks.txt`
- `ghaziabad_house_price_dataset_1000.csv`

### Generated Images

Several `.png` files store visualization outputs generated from the notebooks, mainly from the `train.csv` cleaning and EDA workflow.

## Tech Stack

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Data_Analysis
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Then open any notebook from the browser interface and run the cells in order.

## Suggested Learning Order

If you are using this repository as a learning track, this order fits the current notebook content:

1. `FSML_1.ipynb`
2. `FSML_2.ipynb`
3. `FSML_3.ipynb`
4. `FSML_5_&_6.ipynb`
5. `P_1_FSML_5_&_6.ipynb`
6. `FSML_7.ipynb`
7. `FSML_8.ipynb`
8. `FSML_9.ipynb`

## Notes Before Publishing

- Rename `.gitignore.txt` to `.gitignore` if you want GitHub to apply those ignore rules automatically.
- Large notebook output files and generated images can make the repository heavy; consider clearing notebook outputs before pushing if you want a cleaner repo.
- `ghaziabad_house_price_dataset_1000.csv` is present in the repository but does not appear to be referenced by the current notebooks.

