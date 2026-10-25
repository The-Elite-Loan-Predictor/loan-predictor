"""Form fields and scoring - everything about the model (no Streamlit, no dataset).

The saved model is a scikit-learn Pipeline (preprocessing + estimator) trained on the raw columns, so the app
just feeds it raw values. Form fields are built from the pipeline's `feature_names_in_` plus optional
overrides in `config.FIELD_SPECS`."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config

# ======================================================================
# FORM FIELDS (0/1 -> Yes/No, dependents_* -> one field)
# ======================================================================

_BINARY_ALIASES = {"0": "No", "1": "Yes", "false": "No", "true": "Yes", "n": "No", "y": "Yes"}

# Sensible defaults for common loan columns. Override or extend in config.FIELD_SPECS, e.g.
#   FIELD_SPECS = {"loan_amount": {"type": "number", "min": 0, "max": 700, "default": 130, "step": 1},
#                  "property_area": {"type": "choice", "options": ["Urban", "Rural"], "default": "Urban"}}
_DEFAULT_SPECS: dict[str, dict] = {
    # 0/1 dummy columns -> plain-language dropdowns ("on" is the label for 1, "off" for 0)
    "gender_male": {"label": "Gender", "type": "binary", "on": "Male", "off": "Female", "default": "Male"},
    "married_yes": {"label": "Marital status", "type": "binary", "on": "Married", "off": "Single",
                    "default": "Married"},
    "self_employed_yes": {"label": "Self-employed", "type": "binary", "default": "No",
                          "help": "Does the applicant run their own business?"},
    "credit_history_1.0": {"label": "Credit history", "type": "binary", "on": "Good", "off": "Poor",
                           "default": "Good",
                           "help": "Good = the applicant has repaid past debts as agreed."},
    "education_not_graduate": {"label": "Education", "type": "binary", "on": "Not a graduate",
                               "off": "Graduate", "default": "Graduate"},
    # same, for datasets where the columns are not one-hot encoded
    "gender": {"type": "binary"},
    "married": {"type": "binary"},
    "education": {"type": "binary"},
    "self_employed": {"type": "binary", "default": "No"},
    "credit_history": {"type": "binary"},
    # numbers
    "loan_amount": {"label": "Loan amount", "help": "How much the applicant wants to borrow."},
    "loan_amount_term": {"label": "Loan term (months)", "type": "number", "min": 12, "max": 480,
                         "default": 360, "step": 12,
                         "help": "How long they have to repay, in months (360 months = 30 years)."},
    "applicant_income": {"label": "Applicant's monthly income",
                         "help": "Income per month before deductions."},
    "coapplicant_income": {"label": "Co-applicant's monthly income",
                           "help": "Income of a second person on the loan. Enter 0 if there is none."},
}
_FALLBACK_NUMBER = {"type": "number", "min": 0, "max": 10_000_000, "default": 0, "step": 1}
_NUMERIC_HINTS = ("amount", "income", "term", "installment", "age", "year", "ratio", "score", "balance", "count")


def _guess_spec(col: str) -> dict:
    """Columns like gender_male, married_yes or credit_history_1.0 are 0/1 dummies -> Yes/No dropdowns."""
    low = col.lower()
    if "_" not in low or any(h in low for h in _NUMERIC_HINTS):
        return _FALLBACK_NUMBER
    return {"type": "binary", "default": "Yes" if low.startswith("credit_history") else "No"}


def _as_text(raw) -> str | None:
    """Normalise a cell (str / int / float / NaN) to lower-case text; None when empty."""
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return None
    if isinstance(raw, (int, float, np.number)) and float(raw).is_integer():
        return str(int(raw))
    text = str(raw).strip().lower()
    return text or None


@dataclass(frozen=True)
class ChoiceField:
    """A dropdown. Each option maps a display label to the model column(s) it sets."""
    key: str
    label: str
    options: dict[str, dict[str, float]]
    default: str
    aliases: dict[str, str] = field(default_factory=dict)
    help: str = ""

    @property
    def choices(self) -> list[str]:
        return list(self.options)

    def encode(self, value: str) -> dict[str, float]:
        return dict(self.options[value])

    def parse(self, raw) -> str | None:
        """Match a CSV / editor cell to one of the option labels (case-insensitive)."""
        text = _as_text(raw)
        if text is None:
            return None
        for label in self.options:
            if label.lower() == text:
                return label
        return self.aliases.get(text)


@dataclass(frozen=True)
class NumberField:
    """A numeric input."""
    key: str
    label: str
    min: float
    max: float
    default: float
    step: float
    help: str = ""

    def encode(self, value: float) -> dict[str, float]:
        return {self.key: float(value)}

    def parse(self, raw) -> float | None:
        if raw is None:
            return None
        value = pd.to_numeric(raw, errors="coerce")
        return None if pd.isna(value) else float(value)


Field = ChoiceField | NumberField


def encode_values(fields: list[Field], values: dict) -> dict[str, float]:
    """Turn the user's answers ({field.key: value}) into one row of model columns."""
    row: dict[str, float] = {}
    for f in fields:
        row.update(f.encode(values[f.key]))
    return row


def _pretty(col: str) -> str:
    return config.FIELD_LABELS.get(col, col.replace("_", " ").capitalize())


def _label_order(label: str) -> int:
    m = re.match(r"\d+", label)
    return int(m.group()) if m else 0


def _dep_label(raw: str) -> str:
    """'1.0' -> '1', '3+' -> '3 or more'."""
    m = re.fullmatch(r"\s*(\d+)(?:\.0+)?(\+?)\s*", raw)
    if not m:
        return raw
    number, plus = m.groups()
    return f"{number} or more" if plus else number


def _dependents_field(cols: list[str]) -> ChoiceField:
    prefix_len = len(config.DEPENDENTS_PREFIX) + 1
    zeros = {c: 0 for c in cols}
    options: dict[str, dict[str, float]] = {_dep_label(c[prefix_len:]): {**zeros, c: 1} for c in cols}
    baseline = _dep_label(config.DEPENDENTS_BASELINE)
    # Reference category (drop_first encoding): every dependents column is 0
    options.setdefault(baseline, dict(zeros))
    options = dict(sorted(options.items(), key=lambda kv: _label_order(kv[0])))
    default = baseline if baseline in options else next(iter(options))
    return ChoiceField(key=config.DEPENDENTS_PREFIX, label=config.DEPENDENTS_LABEL,
                       options=options, default=default,
                       help="How many people depend on the applicant's income.")


def _field_from_spec(col: str, spec: dict) -> Field:
    label = config.FIELD_LABELS.get(col) or spec.get("label") or _pretty(col)
    help_text = spec.get("help", "")
    kind = spec.get("type", "number")
    if kind == "binary":
        if "on" in spec and "off" in spec:                       # custom wording, e.g. Male / Female
            on, off = spec["on"], spec["off"]
            options = {on: {col: 1}, off: {col: 0}}
            default = spec.get("default", on)
            aliases = {"1": on, "0": off, "true": on, "false": off, "yes": on, "no": off, "y": on, "n": off}
            return ChoiceField(col, label, options, default if default in options else on,
                               aliases=aliases, help=help_text)
        options = {name: {col: code} for name, code in config.YES_NO.items()}
        default = spec.get("default", "Yes")
        return ChoiceField(col, label, options, default if default in options else next(iter(options)),
                           aliases=dict(_BINARY_ALIASES), help=help_text)
    if kind == "choice":
        raw_options = spec["options"]
        options = {str(v): {col: v} for v in raw_options}
        default = str(spec.get("default", raw_options[0]))
        return ChoiceField(col, label, options, default if default in options else next(iter(options)),
                           help=help_text)
    return NumberField(col, label,
                       min=float(spec.get("min", 0)), max=float(spec.get("max", 10_000_000)),
                       default=float(spec.get("default", 0)), step=float(spec.get("step", 1)),
                       help=help_text)


def _merged_specs() -> dict[str, dict]:
    """Built-in specs, with config.FIELD_SPECS layered on top key by key."""
    specs = {k: dict(v) for k, v in _DEFAULT_SPECS.items()}
    for k, v in getattr(config, "FIELD_SPECS", {}).items():
        specs[k] = {**specs.get(k, {}), **v}
    return specs


def build_fields(columns: list[str]) -> list[Field]:
    """One form field per model input, in model order. The derived installment is not asked for."""
    specs = _merged_specs()
    dep_cols = [c for c in columns if c.lower().startswith(f"{config.DEPENDENTS_PREFIX}_")]
    fields: list[Field] = []
    dependents_added = False
    for col in columns:
        if col == config.INSTALLMENT_COL:
            continue
        if col in dep_cols:
            if not dependents_added:
                fields.append(_dependents_field(dep_cols))
                dependents_added = True
            continue
        fields.append(_field_from_spec(col, specs.get(col) or _guess_spec(col)))
    return fields


# ======================================================================
# SCORING
# ======================================================================

@dataclass
class InstallmentCalculator:
    """Recreates ``estimated_monthly_installment`` = loan_amount / term * factor.

    Only used when the saved model still expects that column. Without the dataset the factor can't be
    measured, so it comes from ``config.INSTALLMENT_FACTOR`` (default 1.0, flagged as approximate).
    """
    enabled: bool = False
    factor: float = 1.0
    reliable: bool = True

    @classmethod
    def from_config(cls) -> "InstallmentCalculator":
        configured = hasattr(config, "INSTALLMENT_FACTOR")
        return cls(enabled=True, factor=float(getattr(config, "INSTALLMENT_FACTOR", 1.0)), reliable=configured)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.enabled and {"loan_amount", "loan_amount_term"} <= set(df.columns):
            term = df["loan_amount_term"].astype(float).replace(0, np.nan)
            df[config.INSTALLMENT_COL] = (df["loan_amount"].astype(float) / term * self.factor).fillna(0.0)
        return df


class LoanPredictor:
    """A saved scikit-learn Pipeline plus the raw column list it expects."""

    def __init__(self, model, columns: list[str], installment: InstallmentCalculator):
        self.model = model
        self.columns = columns
        self.installment = installment

    @property
    def _estimator(self):
        steps = getattr(self.model, "steps", None)
        return steps[-1][1] if steps else self.model

    @property
    def algorithm(self) -> str:
        return type(self._estimator).__name__

    @property
    def has_proba(self) -> bool:
        return hasattr(self.model, "predict_proba")

    def score(self, rows: list[dict[str, float]]) -> np.ndarray:
        """Approval score (0-1) for each row of model columns (see ``encode_values``)."""
        df = self.installment.apply(pd.DataFrame(rows))
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise KeyError(missing)
        return self._positive_proba(df[self.columns])

    def _positive_proba(self, X) -> np.ndarray:
        model = self.model
        if hasattr(model, "predict_proba"):
            classes = list(getattr(model, "classes_", [0, 1]))
            idx = classes.index(config.POSITIVE_CLASS) if config.POSITIVE_CLASS in classes else -1
            return np.asarray(model.predict_proba(X))[:, idx]
        if hasattr(model, "decision_function"):
            return 1 / (1 + np.exp(-np.ravel(model.decision_function(X))))
        return (np.asarray(model.predict(X)) == config.POSITIVE_CLASS).astype(float)

    def feature_weights(self) -> tuple[pd.DataFrame, bool] | None:
        """(table, signed). Linear models give signed weights, tree models give importances. None if unknown."""
        est = self._estimator
        if hasattr(est, "coef_") and np.ndim(est.coef_) == 2 and est.coef_.shape[0] == 1:
            values, label, signed = np.ravel(est.coef_), "Weight", True
        elif hasattr(est, "feature_importances_"):
            values, label, signed = np.ravel(est.feature_importances_), "Importance", False
        else:
            return None
        try:
            names = [n.split("__", 1)[-1] for n in self.model.named_steps["prep"].get_feature_names_out()]
        except Exception:
            return None
        if len(names) != len(values):
            return None
        table = pd.DataFrame({"Feature": names, label: values})
        return table.reindex(table[label].abs().sort_values(ascending=False).index).reset_index(drop=True), signed


# ======================================================================
# LOADING
# ======================================================================

@dataclass
class Artifacts:
    predictor: LoanPredictor
    fields: list[Field]
    model_label: str
    installment_reliable: bool


def load_artifacts(model_path: Path) -> Artifacts:
    model = joblib.load(model_path)

    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ValueError(
            f"`{model_path.name}` does not expose `feature_names_in_`. It must be a full scikit-learn Pipeline "
            "fitted on a DataFrame (re-run the notebook's export cell and copy the new .joblib into models/). "
            "Check with: type(joblib.load(path)) -> sklearn Pipeline.")
    columns = list(names)

    installment = InstallmentCalculator.from_config() if config.INSTALLMENT_COL in columns else InstallmentCalculator()
    predictor = LoanPredictor(model, columns, installment)

    return Artifacts(
        predictor=predictor,
        fields=build_fields(columns),
        model_label=model_path.stem.replace("_", " ").title(),
        installment_reliable=installment.reliable,
    )


def summarize_results(res: pd.DataFrame | None, threshold: float,
                      band: float = config.BORDERLINE_BAND, max_rows: int = 25) -> dict | None:
    if res is None or res.empty:
        return None
    n = len(res)
    approve = int((res["Score"] >= threshold).sum())
    rows = [dict(applicant=a, score=round(float(s), 3),
                 recommendation="Approve" if s >= threshold else "Decline")
            for a, s in zip(res["Applicant"], res["Score"])][:max_rows]
    return dict(applicants=n, approve=approve, decline=n - approve,
                borderline=int(((res["Score"] - threshold).abs() <= band).sum()),
                avg_score=float(res["Score"].mean()), rows=rows)