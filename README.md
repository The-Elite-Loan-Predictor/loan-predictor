# Loan Predictor

A machine learning project that explores loan application data and trains classifiers to predict loan approval from applicant income, credit history, loan details, and demographic information.

The current workflow runs in Jupyter notebooks: data exploration, cleaning and feature engineering, then model training and evaluation. It compares **Logistic Regression**, **Random Forest**, **Decision Tree**, **XGBoost**, **LightGBM**, **SVM**, **KNN**, **Gaussian Naive Bayes**, and **Multinomial Naive Bayes** classifiers. The trained models are exported to `models/` and served by the Streamlit application in `app/app.py`.

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
   ```

   `imbalanced-learn`, `xgboost`, `lightgbm` and `pytest` are listed in `requirements.txt`.

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

Dependency versions are also unpinned, so results may vary between environments.

## Train and export the models

`src/train_models.py` reproduces the notebook 03 pipeline outside of Jupyter and
writes each model to `models/` with the preprocessing it needs, so the app never
has to retrain anything.

```shell
python src/train_models.py
```

It drops `loan_id`, `loan_status`, `total_income` and `balance_income`, makes the
same stratified 80/20 split, fits the `MinMaxScaler` and the
`SelectKBest(f_classif, k=10)` selector on the training set only, trains the
classifiers, and saves each one as a bundle in `models/pipeline/`:

| Key | Contents |
| --- | --- |
| `model` | The fitted classifier. |
| `scaler` | The fitted `MinMaxScaler`. |
| `selector` | The fitted `SelectKBest`. |
| `feature_columns` | The exact column order the model expects. |
| `numeric_columns` | The columns the scaler transforms. |
| `selected_features` | The features the selector kept. |

The bundles live in `models/pipeline/` rather than `models/` because notebook 03
also exports bare estimators into `models/`. Keeping them apart stops the two
from overwriting each other.

| Model | Test accuracy |
| --- | --- |
| Support Vector Machine | 85.37% |
| Logistic Regression | 85.37% |
| Gaussian Naive Bayes | 84.55% |
| Random Forest | 82.93% |
| K-Nearest Neighbors | 80.49% |
| Decision Tree | 73.98% |
| Multinomial Naive Bayes | 69.11% |

## Run the Streamlit app

```shell
streamlit run app/app.py
```

The app loads the bundles from `models/pipeline/`, collects the applicant
details, applies the stored scaler and feature selector, and shows whether the
application is approved or rejected. The classifier can be switched in the
sidebar. A missing model file, an invalid value, or a failed prediction is
reported in the interface instead of raising an unhandled error.

## Tests

```shell
python -m pytest tests/ -v
```

`tests/test_model_integration.py` checks that every exported model loads without
retraining, that the stored preprocessing matches what the notebook produces,
that a known row gets the same prediction through the app and the notebook, and
that invalid input and missing model files are handled gracefully.

## Project structure

```text
loan-predictor/
|-- app/
|   `-- app.py                               # Streamlit prediction app
|-- data/
|   |-- raw_data/loan_data_raw.csv
|   `-- cleaned_data/cleaned_loan_dataset.csv
|-- models/                                 # Exported model bundles (.joblib)
|   `-- pipeline/                           # Bundles: model + scaler + selector
|-- notebooks/
|   |-- 01_explore_data_analysis.ipynb
|   |-- 02_data_cleaning_and_preprocessing.ipynb
|   `-- 03_modelling.ipynb
|-- outputs/reports/Data Exploration/        # EDA report and charts
|-- src/
|   |-- __init__.py
|   |-- train_models.py                      # Train and export the models
|   `-- model_persistence.py                 # Load models and predict
|-- tests/
|   `-- test_model_integration.py            # Model and app integration tests
|-- requirements.txt
`-- README.md
```

The analysis uses pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, xgboost, lightgbm, and Jupyter. Streamlit, joblib and pytest are used by the app and the test suite.
