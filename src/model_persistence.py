"""
Loan prediction model persistence and inference.

Saves the trained estimator together with the complete preprocessing pipeline
(scaler + feature selector) so the Streamlit app can reproduce the exact
transformations used during training - no retraining required at inference time.

The bundles written by src/train_models.py contain:
    model             the fitted classifier
    scaler            fitted MinMaxScaler
    selector          fitted SelectKBest
    feature_columns   the exact column order the model was trained on
    numeric_columns   the columns the scaler transforms
    selected_features the features the selector kept
    target_names      mapping of the encoded target to a human readable label
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Bundles live in their own sub-directory. notebooks/03_modelling.ipynb also
# writes bare estimators (no scaler or selector) into models/ using similar
# file names, so keeping the bundles in models/pipeline/ avoids collisions.
MODELS_DIR = PROJECT_ROOT / "models" / "pipeline"

# Every key a valid bundle must contain.
BUNDLE_KEYS = (
    "model",
    "scaler",
    "selector",
    "feature_columns",
    "numeric_columns",
    "selected_features",
)

# Default model used by the Streamlit app.
DEFAULT_MODEL = "random_forest.joblib"

# Human readable names for the bundled models.
AVAILABLE_MODELS = {
    "random_forest.joblib": "Random Forest",
    "logistic_regression.joblib": "Logistic Regression",
    "decision_tree.joblib": "Decision Tree",
    "support_vector_machine.joblib": "Support Vector Machine",
    "knn.joblib": "K-Nearest Neighbors",
    "gaussian_naive_bayes.joblib": "Gaussian Naive Bayes",
    "multinomial_naive_bayes.joblib": "Multinomial Naive Bayes",
}

# The columns the Streamlit form collects, in the order the model expects them.
FEATURE_COLUMNS = [
    "gender_male",
    "married_yes",
    "dependents_1.0",
    "dependents_2.0",
    "dependents_3.0",
    "self_employed_yes",
    "credit_history_1.0",
    "loan_amount",
    "loan_amount_term",
    "applicant_income",
    "coapplicant_income",
    "estimated_monthly_installment",
]

# Columns scaled with MinMaxScaler during training (numeric features only).
NUMERIC_COLUMNS = [
    "loan_amount",
    "loan_amount_term",
    "applicant_income",
    "coapplicant_income",
    "estimated_monthly_installment",
]

# The columns the Streamlit form collects, with their display labels.
INPUT_FIELDS = {
    "gender_male": "Gender (Male)",
    "married_yes": "Married",
    "dependents_1.0": "Dependents = 1",
    "dependents_2.0": "Dependents = 2",
    "dependents_3.0": "Dependents = 3+",
    "self_employed_yes": "Self Employed",
    "credit_history_1.0": "Credit History (1 = positive)",
    "loan_amount": "Loan Amount",
    "loan_amount_term": "Loan Amount Term (months)",
    "applicant_income": "Applicant Income",
    "coapplicant_income": "Coapplicant Income",
    "estimated_monthly_installment": "Estimated Monthly Installment",
}

# Binary (0/1) inputs that must hold a boolean-like value.
BINARY_FIELDS = [
    "gender_male",
    "married_yes",
    "dependents_1.0",
    "dependents_2.0",
    "dependents_3.0",
    "self_employed_yes",
    "credit_history_1.0",
]


class ModelNotFoundError(FileNotFoundError):
    """Raised when the requested model bundle is missing from models/."""


class PredictionError(RuntimeError):
    """Raised when a prediction cannot be produced."""


def list_available_models():
    """Return the models that are actually present in models/pipeline/."""
    if not MODELS_DIR.exists():
        return []
    return sorted(
        f.name for f in MODELS_DIR.glob("*.joblib") if f.name in AVAILABLE_MODELS
    )


def model_path(filename=DEFAULT_MODEL):
    """Return the path a model bundle is expected at."""
    return MODELS_DIR / filename


def save_pipeline(bundle, filename=DEFAULT_MODEL):
    """Persist a model bundle to models/pipeline/<filename>."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(filename)
    joblib.dump(bundle, path)
    return path


def load_pipeline(filename=DEFAULT_MODEL):
    """Load a persisted model bundle. Raises ModelNotFoundError if missing."""
    path = model_path(filename)
    if not path.exists():
        raise ModelNotFoundError(
            f"Model file '{path}' not found. Run 'python src/train_models.py' "
            "to train and export the models first."
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # pragma: no cover - corrupt/unreadable bundle
        raise PredictionError(f"Could not load model '{filename}': {exc}") from exc

    # A file that is not a bundle (for example a bare estimator exported by the
    # modelling notebook) cannot be used for inference.
    if not isinstance(bundle, dict) or not all(
        key in bundle for key in BUNDLE_KEYS
    ):
        raise PredictionError(
            f"Model file '{filename}' is not a valid model bundle. It must contain "
            f"the model and its preprocessing ({', '.join(BUNDLE_KEYS)}). "
            "Re-run 'python src/train_models.py' to rebuild it."
        )
    return bundle


def _validate_input(row):
    """Validate a single applicant dict and return it as a DataFrame."""
    if not isinstance(row, dict):
        raise PredictionError("Input must be a dictionary of applicant fields.")

    missing = [key for key in FEATURE_COLUMNS if key not in row]
    if missing:
        raise PredictionError(f"Missing required fields: {', '.join(missing)}")

    cleaned = {}
    for key in FEATURE_COLUMNS:
        value = row[key]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise PredictionError(f"Field '{key}' must not be empty.")
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise PredictionError(
                f"Field '{key}' must be a number, got {value!r}."
            ) from exc

    for key in ("loan_amount", "loan_amount_term", "applicant_income"):
        if cleaned[key] <= 0:
            raise PredictionError(f"Field '{key}' must be greater than zero.")

    dependents = (
        cleaned["dependents_1.0"]
        + cleaned["dependents_2.0"]
        + cleaned["dependents_3.0"]
    )
    if dependents > 1:
        raise PredictionError("Select at most one dependents option.")

    for key in BINARY_FIELDS:
        if cleaned[key] not in (0.0, 1.0):
            raise PredictionError(f"Field '{key}' must be 0 or 1.")

    return pd.DataFrame([cleaned])


def predict(row, filename=DEFAULT_MODEL):
    """
    Predict the loan status for a single applicant.

    Args:
        row: dict of applicant fields (see FEATURE_COLUMNS).
        filename: the model bundle to use.

    Returns:
        dict with the raw prediction, the human readable status and the
        probability of approval.
    """
    bundle = load_pipeline(filename)

    frame = _validate_input(row)

    # Reproduce the training transformations exactly.
    numeric_columns = bundle["numeric_columns"]
    scaler = bundle["scaler"]
    selector = bundle["selector"]

    frame = frame[bundle["feature_columns"]]
    frame[numeric_columns] = scaler.transform(frame[numeric_columns])
    selected = selector.transform(frame)

    model = bundle["model"]
    prediction = int(model.predict(selected)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(selected)[0]
        # Target 1 == Approved, so the positive class is index 1.
        probability = float(proba[1])

    status = "Approved" if prediction == 1 else "Rejected"

    return {
        "prediction": prediction,
        "status": status,
        "probability": probability,
        "model": AVAILABLE_MODELS.get(filename, filename),
    }


if __name__ == "__main__":
    print(f"Models directory: {MODELS_DIR}")
    for f in sorted(MODELS_DIR.glob("*.joblib")):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")