"""
Loan Prediction App
===================

Streamlit front-end for the trained loan approval models.

The app loads the model bundles written by src/train_models.py, collects the
applicant details, reproduces the training preprocessing (MinMaxScaler +
SelectKBest) and shows the predicted loan status.

Run from the project root:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make the project root importable so src/ can be used.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_persistence import (  # noqa: E402
    AVAILABLE_MODELS,
    ModelNotFoundError,
    PredictionError,
    list_available_models,
    load_pipeline,
    predict,
)

st.set_page_config(page_title="Loan Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Loan Prediction App")
st.write(
    "Fill in the applicant details below to predict whether the loan "
    "application will be **approved** or **rejected**."
)


@st.cache_data(show_spinner=False)
def get_available_models():
    """Return the model bundles present in models/."""
    return list_available_models()


@st.cache_resource(show_spinner="Loading model…")
def load_model_bundle(filename):
    """Load and cache a model bundle for the lifetime of the app."""
    return load_pipeline(filename)


def main():
    available = get_available_models()

    if not available:
        st.error(
            "No trained models were found in the `models/` folder. "
            "Run `python src/train_models.py` to train and export them first."
        )
        return

    # ---- Model selection -------------------------------------------------
    model_labels = {AVAILABLE_MODELS.get(name, name): name for name in available}
    default_label = AVAILABLE_MODELS.get("random_forest.joblib")
    if default_label not in model_labels:
        default_label = list(model_labels)[0]
    default_index = list(model_labels).index(default_label)

    st.sidebar.header("Model")
    selected_label = st.sidebar.selectbox(
        "Choose a classifier",
        list(model_labels),
        index=default_index,
        help="Each model was trained on the same preprocessed dataset.",
    )
    selected_file = model_labels[selected_label]

    try:
        bundle = load_model_bundle(selected_file)
    except (ModelNotFoundError, PredictionError) as exc:
        st.error(f"Could not load the selected model: {exc}")
        return

    st.sidebar.caption(
        f"Features used: {len(bundle.get('selected_features', []))} (selected by f_classif)"
    )

    # ---- Applicant form --------------------------------------------------
    st.subheader("Applicant Details")

    with st.form("loan_application_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Personal**")
            gender_male = st.selectbox("Gender", ["Male", "Female"])
            married_yes = st.selectbox("Married", ["No", "Yes"])
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
            self_employed_yes = st.selectbox("Self Employed", ["No", "Yes"])

        with col2:
            st.markdown("**Financial**")
            credit_history_1 = st.selectbox("Credit History", ["1 (positive)", "0 (negative)"])
            loan_amount = st.number_input(
                "Loan Amount", min_value=1.0, value=150000.0, step=1000.0
            )
            loan_amount_term = st.number_input(
                "Loan Amount Term (months)", min_value=1.0, value=360.0, step=12.0
            )
            applicant_income = st.number_input(
                "Applicant Income", min_value=1.0, value=5000.0, step=100.0
            )
            coapplicant_income = st.number_input(
                "Coapplicant Income", min_value=0.0, value=0.0, step=100.0
            )

        st.markdown("**Loan Details**")
        default_installment = (
            round(loan_amount / loan_amount_term, 1) if loan_amount_term else 0.0
        )
        estimated_monthly_installment = st.number_input(
            "Estimated Monthly Installment",
            min_value=0.0,
            value=default_installment,
            step=10.0,
            help="Calculated as loan amount / loan term during cleaning.",
        )

        submitted = st.form_submit_button(
            "🔮 Predict Loan Status", use_container_width=True
        )

    if not submitted:
        st.info("Fill in the form and press **Predict Loan Status**.")
        return

    # ---- Build the model input -------------------------------------------
    applicant = {
        "gender_male": 1.0 if gender_male == "Male" else 0.0,
        "married_yes": 1.0 if married_yes == "Yes" else 0.0,
        "dependents_1.0": 1.0 if dependents == "1" else 0.0,
        "dependents_2.0": 1.0 if dependents == "2" else 0.0,
        "dependents_3.0": 1.0 if dependents == "3+" else 0.0,
        "self_employed_yes": 1.0 if self_employed_yes == "Yes" else 0.0,
        "credit_history_1.0": 1.0 if credit_history_1.startswith("1") else 0.0,
        "loan_amount": float(loan_amount),
        "loan_amount_term": float(loan_amount_term),
        "applicant_income": float(applicant_income),
        "coapplicant_income": float(coapplicant_income),
        "estimated_monthly_installment": float(estimated_monthly_installment),
    }

    # ---- Predict ----------------------------------------------------------
    try:
        result = predict(applicant, filename=selected_file)
    except PredictionError as exc:
        st.error(f"Invalid input: {exc}")
        return
    except Exception as exc:  # pragma: no cover - unexpected failure
        st.error(f"Prediction failed: {exc}")
        return

    st.subheader("Prediction Result")

    if result["prediction"] == 1:
        st.success(
            "✅ **Loan Approved** — the application meets the model's criteria."
        )
    else:
        st.error(
            "❌ **Loan Rejected** — the application does not meet the model's criteria."
        )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Model", result["model"])
    col_b.metric("Status", result["status"])
    if result["probability"] is not None:
        col_c.metric("Approval Probability", f"{result['probability'] * 100:.1f}%")
    else:
        col_c.metric("Approval Probability", "n/a")

    with st.expander("Applicant summary"):
        st.json(applicant)


if __name__ == "__main__":
    main()
else:
    main()
