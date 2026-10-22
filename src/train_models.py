"""
Train the loan prediction models and export them for the Streamlit app.

Reproduces the exact pipeline used in notebooks/03_modelling.ipynb:

    1. Load the cleaned dataset
    2. Drop loan_id, loan_status, total_income and balance_income
    3. Stratified train/test split (80/20, random_state=42)
    4. MinMaxScaler on the numeric columns (fit on train only)
    5. SelectKBest(f_classif, k=10) feature selection (fit on train only)
    6. Train the classifiers

Every fitted preprocessing object is bundled with its estimator and written to
models/<name>.joblib so the Streamlit app can reproduce the identical
transformations at inference time without retraining.

Run from the project root:
    python src/train_models.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "cleaned_data" / "cleaned_loan_dataset.csv"

# Bundles live in their own sub-directory. notebooks/03_modelling.ipynb also
# writes bare estimators (no scaler or selector) into models/ using similar
# file names, so keeping the bundles in models/pipeline/ avoids collisions.
MODELS_DIR = PROJECT_ROOT / "models" / "pipeline"

# Columns dropped before training: the identifier, the target, and the two
# engineered features that are derived from other columns.
DROP_COLUMNS = ["loan_id", "loan_status", "total_income", "balance_income"]

# Numeric columns scaled during training. Kept in sync with
# notebooks/03_modelling.ipynb.
NUMERIC_COLUMNS = [
    "loan_amount",
    "loan_amount_term",
    "applicant_income",
    "coapplicant_income",
    "estimated_monthly_installment",
]

# Number of features kept by the f_classif selector.
N_FEATURES = 10

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    """Return the cleaned dataset."""
    return pd.read_csv(DATA_FILE)


def build_models():
    """Return the {display name: estimator} mapping used across the project."""
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(
            max_depth=5,
            class_weight="balanced",
            n_estimators=50,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Support Vector Machine": SVC(
            kernel="linear", gamma="auto", C=1, decision_function_shape="ovr"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Multinomial Naive Bayes": MultinomialNB(),
    }


def train_and_export():
    """Train every model and write it, with its preprocessing, to models/."""
    df = load_data()

    X = df.drop(columns=DROP_COLUMNS)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Scale the numeric columns, fitting the scaler on the training set only.
    scaler = MinMaxScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERIC_COLUMNS] = scaler.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test_scaled[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])

    # Select the most statistically relevant features on the training set only.
    selector = SelectKBest(score_func=f_classif, k=N_FEATURES)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    selected_features = list(X_train_scaled.columns[selector.get_support()])
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"Train shape: {X_train_selected.shape}  Test shape: {X_test_selected.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, model in build_models().items():
        model.fit(X_train_selected, y_train)

        train_accuracy = accuracy_score(y_train, model.predict(X_train_selected))
        test_accuracy = accuracy_score(y_test, model.predict(X_test_selected))

        filename = name.lower().replace(" ", "_") + ".joblib"
        bundle = {
            "model": model,
            "scaler": scaler,
            "selector": selector,
            "feature_columns": list(X.columns),
            "numeric_columns": NUMERIC_COLUMNS,
            "selected_features": selected_features,
            "target_names": {"0": "Rejected", "1": "Approved"},
        }
        joblib.dump(bundle, MODELS_DIR / filename)

        results[name] = (train_accuracy, test_accuracy)
        print(
            f"{name:<26} train {train_accuracy * 100:6.2f}%  "
            f"test {test_accuracy * 100:6.2f}%  -> {filename}"
        )

    print(f"\nSaved {len(results)} model bundles to {MODELS_DIR}")
    return results


if __name__ == "__main__":
    train_and_export()
