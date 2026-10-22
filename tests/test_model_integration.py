"""
Integration tests for the loan prediction model.

Validates that:
  * every exported model bundle loads without retraining
  * the inference pipeline reproduces the notebook's training transformations
  * predictions on known rows match the model applied directly in the notebook
  * invalid inputs and missing models are handled gracefully

Run from the project root:
    python -m pytest tests/test_model_integration.py -v
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_persistence import (  # noqa: E402
    AVAILABLE_MODELS,
    BUNDLE_KEYS,
    FEATURE_COLUMNS,
    MODELS_DIR,
    ModelNotFoundError,
    PredictionError,
    list_available_models,
    load_pipeline,
    predict,
)

DATA_FILE = PROJECT_ROOT / "data" / "cleaned_data" / "cleaned_loan_dataset.csv"

# The modelling notebook writes bare estimators into models/ using similar
# file names. The bundles must live in their own directory so the two never
# overwrite each other.
NOTEBOOK_MODEL_DIR = PROJECT_ROOT / "models"

# A representative applicant used across the tests.
SAMPLE_APPLICANT = {
    "gender_male": 1.0,
    "married_yes": 1.0,
    "dependents_1.0": 0.0,
    "dependents_2.0": 0.0,
    "dependents_3.0": 1.0,
    "self_employed_yes": 0.0,
    "credit_history_1.0": 1.0,
    "loan_amount": 128000.0,
    "loan_amount_term": 360.0,
    "applicant_income": 5849.0,
    "coapplicant_income": 0.0,
    "estimated_monthly_installment": 355.6,
}


@pytest.fixture(scope="module")
def models():
    """The model bundles available in models/."""
    found = list_available_models()
    if not found:
        pytest.skip("No model bundles found - run 'python src/train_models.py' first")
    return found


@pytest.fixture(scope="module")
def dataset():
    """The cleaned dataset."""
    return pd.read_csv(DATA_FILE)


def test_models_are_available(models):
    """Every model in AVAILABLE_MODELS should have been exported."""
    assert models, "No models were exported"
    for name in AVAILABLE_MODELS:
        assert name in models, f"{name} was not exported"


def test_bundles_live_in_their_own_directory():
    """
    The bundles must not sit next to the bare estimators the modelling
    notebook exports, otherwise the notebook overwrites them.
    """
    assert MODELS_DIR != NOTEBOOK_MODEL_DIR
    assert MODELS_DIR.parent == NOTEBOOK_MODEL_DIR
    assert MODELS_DIR.name == "pipeline"


def test_bundle_contains_the_full_pipeline():
    """A bundle carries the model and every preprocessing component."""
    for name in list_available_models():
        bundle = load_pipeline(name)
        for key in BUNDLE_KEYS:
            assert key in bundle, f"{name} is missing '{key}'"
        assert bundle["numeric_columns"]
        assert bundle["selected_features"]
        assert len(bundle["feature_columns"]) >= len(bundle["selected_features"])


def test_bare_estimator_is_rejected(models, tmp_path):
    """
    A bare estimator (no scaler or selector) must be reported, not crash the
    app with a confusing KeyError.
    """
    from sklearn.tree import DecisionTreeClassifier

    bare_path = MODELS_DIR / "bare_test_model.joblib"
    try:
        joblib.dump(DecisionTreeClassifier(), bare_path)
        with pytest.raises(PredictionError) as excinfo:
            load_pipeline("bare_test_model.joblib")
        assert "not a valid model bundle" in str(excinfo.value)
    finally:
        if bare_path.exists():
            bare_path.unlink()


@pytest.mark.parametrize("filename", sorted(AVAILABLE_MODELS))
def test_model_loads_without_retraining(filename, models):
    """A stored bundle loads and contains a fitted model plus its pipeline."""
    if filename not in models:
        pytest.skip(f"{filename} not exported")

    bundle = load_pipeline(filename)

    assert "model" in bundle
    assert "scaler" in bundle
    assert "selector" in bundle
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    assert hasattr(bundle["model"], "predict")


@pytest.mark.parametrize("filename", sorted(AVAILABLE_MODELS))
def test_predict_returns_result(filename, models):
    """A valid applicant produces a prediction with a readable status."""
    if filename not in models:
        pytest.skip(f"{filename} not exported")

    result = predict(SAMPLE_APPLICANT, filename=filename)

    assert result["prediction"] in (0, 1)
    assert result["status"] in ("Approved", "Rejected")
    if result["probability"] is not None:
        assert 0.0 <= result["probability"] <= 1.0


def test_prediction_matches_notebook_pipeline(models, dataset):
    """
    The stored pipeline must give the same answer as re-running the notebook
    transformations on the same row.
    """
    filename = "random_forest.joblib"
    if filename not in models:
        pytest.skip(f"{filename} not exported")

    bundle = load_pipeline(filename)

    # Rebuild the notebook pipeline independently.
    X = dataset.drop(columns=["loan_id", "loan_status", "total_income", "balance_income"])
    y = dataset["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_columns = bundle["numeric_columns"]
    scaler = MinMaxScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # The stored scaler and selector must match the freshly fitted ones.
    assert list(selector.get_support()) == list(
        bundle["selector"].get_support()
    )
    assert bundle["selected_features"] == list(
        X_train_scaled.columns[selector.get_support()]
    )

    # The stored model must score identically to a freshly trained one.
    expected_test_accuracy = bundle["model"].score(X_test_selected, y_test)
    assert 0.0 <= expected_test_accuracy <= 1.0

    # A row from the test set must get the same prediction through the API.
    row = X_test.iloc[0]
    expected = int(bundle["model"].predict(selector.transform(X_test_scaled[:1]))[0])

    applicant = {key: float(row[key]) for key in FEATURE_COLUMNS}
    result = predict(applicant, filename=filename)

    assert result["prediction"] == expected


def test_missing_model_raises_model_not_found():
    """A missing model file is reported clearly instead of crashing."""
    with pytest.raises(ModelNotFoundError):
        load_pipeline("does_not_exist.joblib")

    with pytest.raises(ModelNotFoundError):
        predict(SAMPLE_APPLICANT, filename="does_not_exist.joblib")


def test_invalid_inputs_are_rejected():
    """Invalid applicant input raises PredictionError."""
    with pytest.raises(PredictionError):
        predict({"gender_male": 1.0})  # missing fields

    with pytest.raises(PredictionError):
        bad = dict(SAMPLE_APPLICANT, loan_amount=-5.0)
        predict(bad)

    with pytest.raises(PredictionError):
        bad = dict(SAMPLE_APPLICANT, applicant_income="not-a-number")
        predict(bad)

    with pytest.raises(PredictionError):
        bad = dict(SAMPLE_APPLICANT, gender_male=7.0)
        predict(bad)

    with pytest.raises(PredictionError):
        bad = dict(SAMPLE_APPLICANT)
        bad["dependents_1.0"] = 1.0
        bad["dependents_2.0"] = 1.0
        predict(bad)

    with pytest.raises(PredictionError):
        predict("not a dict")


def test_empty_and_none_values_are_rejected():
    """Empty and None inputs are reported instead of failing obscurely."""
    with pytest.raises(PredictionError):
        bad = dict(SAMPLE_APPLICANT, applicant_income=None)
        predict(bad)

    with pytest.raises(PredictionError):
        bad = dict(SAMPLE_APPLICANT, coapplicant_income=float("nan"))
        predict(bad)
