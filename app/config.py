"""Configurations: every tunable value lives here."""
from __future__ import annotations

import os
from pathlib import Path

# --- Branding ---------------------------------------------------------------------------
APP_TITLE = "Loan Predictor"
APP_ICON = "🏦"
TEAM_NAME = "The Elite Loan Predictors"
SLOGAN = "Elite decisions. Smarter lending."
HERO_TEXT = "Every decision scored. Every loan reviewed. Lend with confidence."   # line under the slogan

# --- Paths ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "gaussian_naive_bayes.joblib"))

MODEL_NAME = ""

# --- Backend table of assessed applications ---------------------------------------------
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_ROOT / "app" / "loan_applications.db"))



# --- Model inputs -----------------------------------------------------------------------
INSTALLMENT_COL = "estimated_monthly_installment"   

INSTALLMENT_FACTOR = 1.0
POSITIVE_CLASS = 1                   

# --- Decision policy --------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.50                
BORDERLINE_BAND = 0.10                

# --- Form presentation ------------------------------------------------------------------
YES_NO = {"No": 0, "Yes": 1}           
DEPENDENTS_PREFIX = "dependents"       
DEPENDENTS_LABEL = "Number of dependents"
DEPENDENTS_BASELINE = "0"           
FIELD_LABELS: dict[str, str] = {}      


FIELD_SPECS: dict[str, dict] = {
    "loan_amount": {"type": "number", "min": 0, "max": 150000, "default": 12000, "step": 1},
    "loan_amount_term": {"type": "number", "min": 12, "max": 480, "default": 360, "step": 12},
    "applicant_income": {"type": "number", "min": 0, "max": 150000, "default": 8000, "step": 100},
    "coapplicant_income": {"type": "number", "min": 0, "max": 150000, "default": 8000, "step": 100},
}

# --- AI assistant (OpenRouter) ----------------------------------------------------------
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
ASSISTANT_MAX_TOKENS = 700
ASSISTANT_HISTORY = 12               
