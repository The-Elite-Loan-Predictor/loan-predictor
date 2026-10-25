"""AI assistant: OpenRouter"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator

import streamlit as st

import config

try:                                
    from openai import OpenAI
except ImportError:
    OpenAI = None


SYSTEM_PROMPT = """You are the assistant inside {team}'s loan-prediction app, used by lending staff.

Use only the CONTEXT JSON below (current single assessment, summary and rows of the applicants table,
threshold, model name and accuracy). If something is not in it, say you don't have it - never invent
numbers.

Be concise, friendly and use plain business language. The score is decision support, not a final
decision; a human must make the final call. Never advise using protected characteristics (e.g.
gender, marital status) to justify a decision.

CONTEXT:
{context}
"""

SUGGESTIONS = [
    "Why was this decision made?",
    "What does the score mean?",
    "How does the threshold work?",
    "How accurate is the model?",
]


def get_openrouter_api_key() -> str:
    """Read the key from Streamlit secrets or environment variables."""
    try:
        key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("OPENROUTER_API_KEY", "")


def get_openrouter_client(api_key: str | None = None):
    """Create an OpenRouter client, or return None if the package or key is missing."""
    if OpenAI is None:
        return None

    key = api_key or get_openrouter_api_key()

    if not key:
        return None

    return OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
    )


def llm_available(api_key: str | None = None) -> bool:
    return OpenAI is not None and bool(api_key or get_openrouter_api_key())


def stream_llm(
    api_key: str | None,
    history: list[dict],
    context: dict,
) -> Iterator[str]:
    client = get_openrouter_client(api_key)

    if client is None:
        raise RuntimeError(
            "OpenRouter is not available. Install `openai` and configure OPENROUTER_API_KEY."
        )

    model = (
        os.getenv("OPENROUTER_MODEL")
        or getattr(config, "OPENROUTER_MODEL", "")
        or "openai/gpt-4o-mini"
    )

    system = SYSTEM_PROMPT.format(
        team=getattr(config, "TEAM_NAME", "the loan assessment team"),
        context=json.dumps(context, indent=2, default=str),
    )

    messages = [{"role": "system", "content": system}]

    for item in history[-getattr(config, "ASSISTANT_HISTORY", 12):]:
        role = item.get("role")
        content = item.get("content")

        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=getattr(config, "ASSISTANT_MAX_TOKENS", 700),
        temperature=0.3,
        stream=True,
    )

    for chunk in response:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if isinstance(content, str) and content:
            yield content


def rule_based_reply(question: str, ctx: dict) -> str:
    """Provide useful predefined answers without an API key."""
    q = question.lower()
    assessment = ctx.get("assessment")
    threshold = ctx.get("threshold", config.DEFAULT_THRESHOLD)
    band = int(config.BORDERLINE_BAND * 100)

    if any(word in q for word in (
        "table", "batch", "applicants", "summar", "portfolio"
    )):
        table = ctx.get("applicants_table")

        if not table:
            return (
                "No applicants have been assessed in the table yet. "
                "Use the Applicants table page first."
            )

        return (
            f"**{table['applicants']}** applicants assessed: "
            f"**{table['approve']}** approve and "
            f"**{table['decline']}** decline "
            f"(**{table['borderline']}** borderline — consider manual review). "
            f"Average score: **{table['avg_score']:.0%}**. "
            f"Threshold: **{threshold:.0%}**."
        )

    if any(word in q for word in ("threshold", "policy", "cut")):
        return (
            f"Applications scoring **{threshold:.0%} or higher** are "
            "recommended for approval; lower scores are recommended for decline. "
            f"Scores within {band} percentage points of the threshold are "
            "flagged as borderline for manual review. You can change the "
            "threshold in the sidebar."
        )

    if any(word in q for word in ("accura", "reliable", "trust")):
        accuracy = ctx.get("model_test_accuracy")

        if accuracy is None:
            return "I don't have an accuracy figure for this model."

        return (
            f"**{ctx.get('model', 'The model')}** scores "
            f"**{accuracy:.0%}** on the held-out test set. "
            "Accuracy alone does not establish fairness or reliability. "
            "Use the model as decision support, not a guarantee."
        )

    if any(word in q for word in ("mean", "score", "probab")):
        return (
            "The score represents the model's estimated probability of the "
            "positive outcome defined by your model. Check how your target "
            "variable is encoded to confirm whether that means loan approval. "
            "Treat the score as decision support, not a guarantee."
        )

    if any(word in q for word in ("why", "explain", "reason", "decision")):
        if not assessment:
            return (
                "Run an assessment on the Assess applicant page first, "
                "then ask me again."
            )

        inputs = assessment.get("inputs", {})
        details = ", ".join(
            f"{key}: {value}" for key, value in inputs.items()
        )

        return (
            f"The model scored this application **{assessment['score']:.0%}** "
            f"against a threshold of **{threshold:.0%}**. "
            f"The recommendation is **{assessment['decision']}**.\n\n"
            f"Inputs used: {details}.\n\n"
            "This describes the score and threshold, not the causal reasons "
            "behind the model's prediction."
        )

    return (
        "I can explain the latest decision"
        "say what the score means, explain model accuracy, or describe how "
        "the threshold works. Configure an OpenRouter API key for free-form "
        "AI answers."
    )