# Loan Predictor

A machine learning project that explores loan application data and trains classifiers to predict loan approval from applicant income, credit history, loan details, and demographic information.

The current workflow runs in Jupyter notebooks: data exploration, cleaning and feature engineering, then model training and evaluation. It compares **Logistic Regression**, **Random Forest**, and **Decision Tree** classifiers. The Streamlit application is planned; `app/app.py` is currently an empty placeholder.

## Setup

Install Python with `pip`, clone or download this repository, and open a terminal in the project root.

1. Create a virtual environment:

   ```shell
   python -m venv .venv
   ```

2. Activate it using the command for your shell:

   **Windows PowerShell**

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **Windows Command Prompt**

   ```bat
   .venv\Scripts\activate.bat
   ```

   **macOS / Linux**

   ```bash
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```shell
   python -m pip install -r requirements.txt
   python -m pip install imbalanced-learn
   ```

   The modelling notebook imports `RandomOverSampler` from `imblearn`. Its package, `imbalanced-learn`, is not yet listed in `requirements.txt`, so the second command is required.

## Run the notebooks

With the virtual environment active, start Jupyter from the `notebooks` directory:

```shell
cd notebooks
python -m notebook
```

Open each notebook in the following order and run its cells from top to bottom:

| Notebook | Purpose |
| --- | --- |
| [01_explore_data_analysis.ipynb](notebooks/01_explore_data_analysis.ipynb) | Inspect the raw dataset, missing values, duplicates, summary statistics, and categorical distributions. |
| [02_data_cleaning_and_preprocessing.ipynb](notebooks/02_data_cleaning_and_preprocessing.ipynb) | Clean and encode the data, engineer features, and export the processed CSV. |
| [03_modelling.ipynb](notebooks/03_modelling.ipynb) | Split the data, scale numeric features, select features, train three classifiers, and evaluate predictions. |

The notebooks use paths relative to `notebooks/`, such as `../data/raw_data/loan_data_raw.csv`. If you use another notebook editor, ensure its kernel uses this working directory and the virtual environment created above.

Both datasets are included in the repository. Running notebook 02 overwrites `data/cleaned_data/cleaned_loan_dataset.csv`; notebook 03 reads that file.

## Dataset

The [raw dataset](data/raw_data/loan_data_raw.csv) contains **614 applications and 13 columns**:

| Group | Columns |
| --- | --- |
| Identifier | `Loan_ID` |
| Applicant details | `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed` |
| Income | `ApplicantIncome`, `CoapplicantIncome` |
| Loan and credit details | `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area` |
| Target | `Loan_Status`: `Y` (approved) or `N` (rejected) |

The [processed dataset](data/cleaned_data/cleaned_loan_dataset.csv) contains **614 rows and 16 columns**, with `loan_status` encoded as `1` for approved and `0` for rejected.

See the [data exploration report](outputs/reports/Data%20Exploration/EDA_Documentation.md) for missing-value counts and distribution charts.

## Preprocessing and feature engineering

Notebook 02 performs the following steps:

- Standardizes column names and converts the `3+` dependents category to `3`.
- Fills missing gender, marital status, dependents, self-employment, and credit-history values with the most frequent value; fills missing loan amounts and terms with the median.
- One-hot encodes the selected categorical fields and label-encodes the target.
- Creates `total_income` from applicant and coapplicant income, and multiplies the raw loan amount by 1,000.
- Calculates `estimated_monthly_installment` as loan amount divided by loan term, rounded to one decimal place, and `balance_income` as total income minus that installment. This installment feature does not include interest or fees.

`Education` and `Property_Area` are present in the raw data but are not included in the current processed dataset.

## Modelling and evaluation

Notebook 03 drops `loan_id`, `total_income`, and `balance_income` from the input features and uses `loan_status` as the target. It then:

1. Creates a stratified **80% training / 20% test** split with `random_state=42`.
2. Fits `StandardScaler` on the training data for five numeric features and transforms both sets.
3. Fits `SelectKBest(f_classif, k=5)` on the training data to select five features.
4. Trains the following models:

   | Model | Configuration |
   | --- | --- |
   | Logistic Regression | Random oversampling of the training set; `max_iter=1000`. |
   | Random Forest | 50 trees, maximum depth of 5, and balanced class weights. |
   | Decision Tree | Gini criterion, maximum depth of 5, and balanced class weights. |

Each model uses `random_state=42`. Evaluation prints training and test accuracy, a classification report with precision, recall, and F1-score, and a confusion matrix. Run the notebook to generate the results; the committed modelling notebook does not contain saved metrics.

### Current limitations

Imputation and categorical encoding are fitted on the full dataset in notebook 02, before the train/test split in notebook 03. Scaling, feature selection, and oversampling use only training data, but the earlier preprocessing can still leak information from the test set. Moving all fitted preprocessing into a training pipeline is a remaining improvement.

Trained models and preprocessing objects remain in notebook memory; model export, an interactive prediction app, and automated tests have not yet been implemented. Dependency versions are also unpinned, so results may vary between environments.

## Project structure

```text
loan-predictor/
|-- app/
|   `-- app.py                               # Placeholder for the Streamlit app
|-- data/
|   |-- raw_data/loan_data_raw.csv
|   `-- cleaned_data/cleaned_loan_dataset.csv
|-- models/                                 # Reserved for saved models
|-- notebooks/
|   |-- 01_explore_data_analysis.ipynb
|   |-- 02_data_cleaning_and_preprocessing.ipynb
|   `-- 03_modelling.ipynb
|-- outputs/reports/Data Exploration/        # EDA report and charts
|-- src/                                    # Placeholder for reusable code
|-- tests/                                  # Placeholder for automated tests
|-- requirements.txt
`-- README.md
```

The analysis uses pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, and Jupyter. Streamlit and joblib are included in the dependencies for future app and model-persistence work.
