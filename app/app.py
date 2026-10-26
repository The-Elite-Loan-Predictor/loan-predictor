from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import assistant
import config
import loandb as store
from model import (
    Artifacts,
    ChoiceField,
    Field,
    encode_values,
    load_artifacts,
)


st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
)


# ======================================================================
# STYLING
# ======================================================================

_CSS = r"""
<style>
@import url("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css");

.block-container {padding-top: 1.6rem; max-width: 1150px;}

.hero {
    background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #38bdf8 100%);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    color: #fff;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(37,99,235,.25);
}
.hero .team {
    display: inline-block;
    background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.35);
    padding: .2rem .85rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .09em;
    text-transform: uppercase;
    margin-bottom: .7rem;
}
.hero .team i {margin-right: .5rem; color: #fde68a;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero h1 i {margin-right: .6rem;}
.hero .tagline {
    margin: .4rem 0 0;
    font-size: 1.25rem;
    font-weight: 700;
    opacity: 1;
    letter-spacing: .01em;
}
.hero p {margin: .35rem 0 0; opacity: .92; font-size: 1.02rem;}

.sec {font-size: 1.45rem; font-weight: 700; margin: .6rem 0 .7rem;}
.sec i {color: #2563eb; margin-right: .55rem;}

.result {
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: .8rem;
    border: 1px solid transparent;
}
.result .label {
    font-size: .85rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    opacity: .75;
}
.result .value {font-size: 2.1rem; font-weight: 800; line-height: 1.2;}
.result .value i {margin-right: .6rem;}
.result .sub {margin-top: .25rem; opacity: .85;}
.r-approve {background: rgba(34,197,94,.14); border-color: rgba(34,197,94,.45);}
.r-approve .value i {color: #16a34a;}
.r-decline {background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.45);}
.r-decline .value i {color: #dc2626;}
.r-border {background: rgba(245,158,11,.14); border-color: rgba(245,158,11,.5);}
.r-border .value i {color: #d97706;}

.app-footer {
    text-align: center;
    opacity: .65;
    font-size: .85rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(100,116,139,.25);
}
.app-footer i {color: #d97706; margin-right: .4rem;}

div[data-testid="stMetric"] {
    background: rgba(100,116,139,.10);
    border-radius: 14px;
    padding: .8rem 1rem;
}
section[data-testid="stSidebar"] .stRadio label {font-size: 1.02rem;}
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
}

/* Font Awesome icons on Streamlit widgets */
.stFormSubmitButton button p::before,
.stDownloadButton button p::before,
.st-key-clear_chat button p::before,
.st-key-save_reviews button p::before,
.st-key-delete_applicants button p::before,
.st-key-nav_assess button p::before,
.st-key-nav_review button p::before,
.st-key-nav_reporting button p::before,
.st-key-nav_assistant button p::before {
    font-family: "Font Awesome 6 Free";
    font-weight: 900;
    margin-right: .6rem;
}
.stFormSubmitButton button p::before {content: "\f002";}
.stDownloadButton button p::before {content: "\f019";}
.st-key-clear_chat button p::before {content: "\f1f8";}
.st-key-save_reviews button p::before {content: "\f0c7";}
.st-key-delete_applicants button p::before {content: "\f1f8";}
.st-key-nav_assess button p::before {content: "\f573";}
.st-key-nav_review button p::before {content: "\f46c";}
.st-key-nav_reporting button p::before {content: "\f200";}
.st-key-nav_assistant button p::before {content: "\f544";}

/* sidebar navigation: full-width, left-aligned buttons */
section[data-testid="stSidebar"] [class*="st-key-nav_"] button {justify-content: flex-start;}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button div {justify-content: flex-start; width: 100%;}
section[data-testid="stSidebar"] [class*="st-key-nav_"] button p {text-align: left;}


/* Chat avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: flex;
    align-items: center;
    justify-content: center;
}
[data-testid="stChatMessageAvatarUser"] svg,
[data-testid="stChatMessageAvatarAssistant"] svg {
    display: none;
}
[data-testid="stChatMessageAvatarUser"]::before,
[data-testid="stChatMessageAvatarAssistant"]::before {
    font-family: "Font Awesome 6 Free";
    font-weight: 900;
    font-size: 1rem;
    color: #fff;
}
[data-testid="stChatMessageAvatarUser"]::before {content: "\f007";}
[data-testid="stChatMessageAvatarAssistant"]::before {content: "\f544";}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# page -> (font family, Google Fonts query). Pages not listed keep the default font.
_PAGE_FONTS = {
    "Loan review": ("IBM Plex Sans", "IBM+Plex+Sans:wght@400;500;600;700"),
    "Reporting": ("Poppins", "Poppins:wght@400;500;600;700"),
    "AI assistant": ("Nunito", "Nunito:wght@400;500;600;700"),
}


def apply_page_font(page: str) -> None:
    """Use this page's font for the main area (the hero banner, sidebar and icons keep theirs)."""
    spec = _PAGE_FONTS.get(page)
    if not spec:
        return
    family, query = spec
    st.markdown(
        f"""<style>
@import url("https://fonts.googleapis.com/css2?family={query}&display=swap");
.block-container :where(div, p, span, label, h1, h2, h3, h4, h5, h6, li, a, button, input, textarea):not(.hero *):not(.fa-solid):not([data-testid="stIconMaterial"]) {{
    font-family: "{family}", sans-serif !important;
}}
</style>""",
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="team">
                <i class="fa-solid fa-trophy"></i>
                {html.escape(config.TEAM_NAME)}
            </div>
            <h1>
                <i class="fa-solid fa-building-columns"></i>
                {html.escape(config.APP_TITLE)}
            </h1>
            <p class="tagline">{html.escape(config.SLOGAN)}</p>
            <p>{html.escape(getattr(config, "HERO_TEXT", "Every decision scored. Every loan reviewed. Lend with confidence."))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def heading(icon: str, text: str) -> None:
    """Section heading with a Font Awesome icon."""
    st.markdown(
        f'<div class="sec"><i class="fa-solid fa-{icon}"></i>'
        f'{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        '<div class="app-footer">'
        '<i class="fa-solid fa-trophy"></i>'
        f'<b>{html.escape(config.TEAM_NAME)}</b> &middot; '
        f'{html.escape(config.SLOGAN)}</div>',
        unsafe_allow_html=True,
    )


# ======================================================================
# COMPONENTS
# ======================================================================

def gauge(score: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 46}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#1e293b", "thickness": 0.22},
                "steps": [
                    {
                        "range": [0, threshold * 100],
                        "color": "#fecaca",
                    },
                    {
                        "range": [threshold * 100, 100],
                        "color": "#bbf7d0",
                    },
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 4},
                    "thickness": 0.9,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=270, margin=dict(l=25, r=25, t=25, b=5))
    return fig


def result_card(decision: str, borderline: bool) -> None:
    css = (
        "r-border"
        if borderline
        else ("r-approve" if decision == "Approve" else "r-decline")
    )

    icon = (
        "triangle-exclamation"
        if borderline
        else "circle-check"
        if decision == "Approve"
        else "circle-xmark"
    )

    sub = (
        "Close to the threshold - consider a manual review."
        if borderline
        else "Score is above your approval threshold."
        if decision == "Approve"
        else "Score is below your approval threshold."
    )

    st.markdown(
        f'<div class="result {css}">'
        '<div class="label">Model recommendation</div>'
        f'<div class="value"><i class="fa-solid fa-{icon}"></i>'
        f'{html.escape(decision)}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# ======================================================================
# SIDEBAR
# ======================================================================

# Key of each page's navigation button (the CSS above gives each one its Font Awesome icon)
_NAV_KEYS = {
    "Assess applicant": "nav_assess",
    "Loan review": "nav_review",
    "Reporting": "nav_reporting",
    "AI assistant": "nav_assistant",
}


def _go(name: str) -> None:
    st.session_state["page"] = name


def render_sidebar(art: Artifacts, pages: list[str]) -> tuple[str, float]:
    sb = st.sidebar

    sb.markdown(
        '<h2><i class="fa-solid fa-building-columns" '
        'style="color:#2563eb"></i>&nbsp; '
        f'{html.escape(config.APP_TITLE)}</h2>',
        unsafe_allow_html=True,
    )

    sb.caption(
        f"<i class='fa-solid fa-trophy'></i>&nbsp; "
        f"**{html.escape(config.TEAM_NAME)}**  \n"
        f"*{html.escape(config.SLOGAN)}*",
        unsafe_allow_html=True,
    )

    page = st.session_state.get("page")
    if page not in pages:
        page = pages[0]
        st.session_state["page"] = page

    for i, name in enumerate(pages):
        sb.button(
            name,
            key=_NAV_KEYS.get(name, f"nav_{i}"),
            use_container_width=True,
            type="primary" if name == page else "secondary",
            on_click=_go,
            args=(name,),
        )

    sb.divider()

    threshold = sb.slider(
        "Approval threshold", 0.05, 0.95, config.DEFAULT_THRESHOLD, 0.05,
        help=(
            "Scores at or above this are recommended for approval. "
            "Raise it to be stricter, lower it to be more lenient. "
            "It applies to new assessments; saved ones keep the threshold used at the time."
        ),
    )
    sb.divider()

    sb.caption(
        f"<i class='fa-solid fa-cube'></i>&nbsp; "
        f"**Model:** {html.escape(art.model_label)}  \n"
        f"<i class='fa-solid fa-microchip'></i>&nbsp; "
        f"**Algorithm:** {html.escape(art.predictor.algorithm)}",
        unsafe_allow_html=True,
    )

    if not art.predictor.has_proba:
        sb.info(
            "This model has no `predict_proba`, so the score is a sigmoid "
            "of its decision function - good for ranking, not a calibrated "
            "probability. Retrain with `SVC(probability=True)`."
        )

    sb.caption(
        "<i class='fa-solid fa-scale-balanced'></i>&nbsp; "
        "Decision support only - a human makes the final call.",
        unsafe_allow_html=True,
    )

    return page, threshold


# ======================================================================
# PAGE 1 - ASSESS ONE APPLICANT
# ======================================================================

def _widget(f):
    """Render the appropriate widget for a model field."""
    if isinstance(f, ChoiceField):
        options = list(f.choices)
        index = options.index(f.default) if f.default in options else 0
        return st.selectbox(
            f.label, options=options, index=index, key=f"field_{f.key}", help=getattr(f, "help", "") or None
        )

    number = float(f.default)
    integer_field = all(
        float(v).is_integer() for v in (f.min, f.max, f.step, f.default)
    )
    if integer_field:
        return st.number_input(
            f.label,
            min_value=int(f.min),
            max_value=int(f.max),
            value=int(round(number)),
            step=int(f.step),
            key=f"field_{f.key}",
            help=getattr(f, "help", "") or None,
        )
    return st.number_input(
        f.label,
        min_value=float(f.min),
        max_value=float(f.max),
        value=number,
        step=float(f.step),
        key=f"field_{f.key}",
        help=getattr(f, "help", "") or None,
    )


def page_assess(art: Artifacts, threshold: float) -> None:
    fields = art.fields
    heading("user-pen", "Applicant details")

    with st.form("applicant"), st.container(border=True):
        id_number = st.text_input(
            "ID number",
            key="field_id_number",
            max_chars=13,
            help="13 characters. Used as the applicant's reference; each ID number can only be assessed once.",
        )
        full_name = st.text_input("Full name", key="field_full_name")
        cols = st.columns(3)
        values: dict = {}

        for i, field in enumerate(fields):
            with cols[i % 3]:
                values[field.key] = _widget(field)

        submitted = st.form_submit_button("Assess application", type="primary")

    if submitted:
        st.session_state.pop("assessment", None)
        id_number = "".join(id_number.split())
        full_name = " ".join(full_name.split())

        if len(id_number) != 13 or not id_number.isalnum():
            st.error(
                "The ID number must be exactly 13 characters "
                f"(letters and digits only, no spaces). You entered {len(id_number)}."
            )
            return

        if len(full_name.split()) < 2:
            st.error("Enter the applicant's full name (first name and surname).")
            return

        if store.exists(id_number):
            st.error(
                f"ID number **{id_number}** has already been assessed. "
                "Each applicant can only be assessed once."
            )
            return

        # Manual-review safeguard for zero combined income.
        income_keys = ("applicant_income", "coapplicant_income")
        if all(key in values for key in income_keys):
            try:
                combined_income = sum(float(values[key]) for key in income_keys)
            except (TypeError, ValueError):
                combined_income = None

            if combined_income is not None and combined_income <= 0:
                st.warning(
                    "The applicant and co-applicant have no positive combined "
                    "income. This application requires manual review."
                )
                return

        try:
            score = float(art.predictor.score([encode_values(fields, values)])[0])
        except KeyError as e:
            st.error(
                "This model expects columns the app doesn't provide: "
                f"`{e.args[0]}`."
            )
            return
        except Exception as e:
            st.error(f"Scoring failed: {e!r}")
            return

        decision = "Approve" if score >= threshold else "Decline"
        borderline = abs(score - threshold) <= config.BORDERLINE_BAND
        shown = {f.label: values[f.key] for f in fields}

        if not store.add(id_number, full_name, score, threshold, decision, art.model_label, shown):
            st.error(f"ID number **{id_number}** has already been assessed.")
            return

        st.session_state["assessment"] = {
            "id_number": id_number,
            "full_name": full_name,
            "inputs": shown,
            "score": score,
            "threshold": threshold,
            "decision": decision,
            "borderline": borderline,
        }

    a = st.session_state.get("assessment")

    if a is None:
        st.info(
            "Fill in the applicant's details and press "
            "**Assess application**."
        )
        return

    st.success(
        f"**{a['full_name']}** (ID {a['id_number']}) was assessed and sent to the "
        "**Loan review** table."
    )

    heading("scale-balanced", "Recommendation")
    left, right = st.columns([1.1, 1])

    with left:
        result_card(a["decision"], a["borderline"])
        m1, m2 = st.columns(2)
        m1.metric("Approval score", f"{a['score']:.0%}")
        m2.metric("Threshold used", f"{a['threshold']:.0%}")

    with right:
        st.plotly_chart(gauge(a["score"], a["threshold"]))

    report = pd.DataFrame(
        [{
            "id_number": a["id_number"],
            "full_name": a["full_name"],
            **a["inputs"],
            "score": round(a["score"], 4),
            "threshold": a["threshold"],
            "recommendation": a["decision"],
            "model": art.model_label,
        }]
    )

    st.download_button(
        "Download this assessment (CSV)",
        report.to_csv(index=False),
        "loan_assessment.csv",
        "text/csv",
    )

    st.caption(
        "<i class='fa-solid fa-comments'></i>&nbsp; "
        "Ask the **AI assistant** to explain this result.",
        unsafe_allow_html=True,
    )

    weights = art.predictor.feature_weights()

    if weights:
        table, signed = weights
        value_col = table.columns[1]

        with st.expander("What drives the score?"):
            st.caption(
                (
                    "Positive weight pushes towards **approve**, negative "
                    "towards **decline**. Inputs are scaled, so weights are "
                    "comparable. "
                    if signed
                    else "Importance shows how much the model relies on each "
                    "feature (not the direction). "
                )
                + "A feature near zero will barely change the score."
            )

            st.dataframe(
                table.assign(**{value_col: table[value_col].round(3)}),
                hide_index=True,
            )


# ======================================================================
# PAGE 2 - LOAN REVIEW (admins)
# ======================================================================

_DECISIONS = ["Pending", "Approved", "Declined"]


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _admin_ok() -> bool:
    """Optional password gate: set ADMIN_PASSWORD in config.py to enable it."""
    password = getattr(config, "ADMIN_PASSWORD", None)

    if not password or st.session_state.get("admin_ok"):
        return True

    entered = st.text_input("Admin password", type="password")

    if entered:
        if entered == password:
            st.session_state["admin_ok"] = True
            st.rerun()
        st.error("Incorrect password.")

    return False


def _do_delete(where: str, labels: dict) -> None:
    """Button callback: delete the selected applicants and reset the controls."""
    picked = st.session_state.get(f"del_pick_{where}", [])
    deleted = store.delete([labels[p] for p in picked if p in labels])
    st.session_state[f"del_pick_{where}"] = []
    st.session_state[f"del_sure_{where}"] = False
    st.session_state["review_ver"] = st.session_state.get("review_ver", 0) + 1
    st.session_state["delete_msg"] = deleted


def _delete_panel(df: pd.DataFrame, where: str) -> None:
    """Pick applicants and permanently delete them (asks for confirmation first)."""
    with st.expander("Delete applicants"):
        if not _admin_ok():
            return

        labels = {
            f"{name} ({id_no or 'no ID'})": int(row_id)
            for name, id_no, row_id in zip(df["applicant"], df["id_number"], df["id"])
        }

        picked = st.multiselect(
            "Select applicants to delete", list(labels), key=f"del_pick_{where}"
        )
        sure = st.checkbox(
            "I understand this permanently removes the selected applicants and their review "
            "decisions, and their ID number can then be assessed again.",
            key=f"del_sure_{where}",
        )
        st.button(
            "Delete selected",
            key="delete_applicants",
            disabled=not (picked and sure),
            on_click=_do_delete,
            args=(where, labels),
        )


def _delete_notice() -> None:
    deleted = st.session_state.pop("delete_msg", None)
    if deleted is not None:
        st.success(f"Deleted {deleted} applicant(s).")


def page_review(art: Artifacts, threshold: float) -> None:
    heading("clipboard-check", "Loan review")

    if not _admin_ok():
        return

    _delete_notice()

    df = store.fetch_all()

    if df.empty:
        st.info("No applications yet. Assessed applicants will appear here for review.")
        return

    everyone = df

    status = st.radio(
        "Show",
        ["All", "Awaiting review", "Approved", "Declined"],
        horizontal=True,
    )

    if status == "Awaiting review":
        df = df[df["final_decision"] == "Pending"]
    elif status in ("Approved", "Declined"):
        df = df[df["final_decision"] == status]

    if df.empty:
        st.info("Nothing to show for this filter.")
        return

    table = pd.DataFrame(
        {
            "_id": df["id"],
            "ID number": df["id_number"],
            "Full name": df["applicant"],
            "Submitted": df["submitted_at"],
            "Score (%)": (df["score"] * 100).round(1),
            "Model recommendation": df["model_decision"],
            "Final decision": df["final_decision"],
            "Admin note": df["admin_note"],
        }
    ).reset_index(drop=True)

    saved = st.session_state.pop("review_saved", None)
    if saved is not None:
        st.success(f"Saved {saved} change(s).")

    ver = st.session_state.setdefault("review_ver", 0)

    edited = st.data_editor(
        table,
        hide_index=True,
        num_rows="fixed",
        disabled=["_id", "ID number", "Full name", "Submitted", "Score (%)", "Model recommendation"],
        column_config={
            "_id": None,
            "Final decision": st.column_config.SelectboxColumn(
                "Final decision", options=_DECISIONS, required=True,
                help="Pending until an admin approves or declines the loan.",
            ),
            "Admin note": st.column_config.TextColumn("Admin note"),
        },
        key=f"review_editor_{ver}_{status}",
    )

    if st.button("Save decisions", type="primary", key="save_reviews"):
        changed = 0

        for old, new in zip(table.to_dict("records"), edited.to_dict("records")):
            new_note = _text(new["Admin note"])
            if (
                old["Final decision"] != new["Final decision"]
                or _text(old["Admin note"]) != new_note
            ):
                store.update_review(old["_id"], new["Final decision"], new_note)
                changed += 1

        st.session_state["review_saved"] = changed
        st.session_state["review_ver"] = ver + 1
        st.rerun()

    with st.expander("View application details"):
        labels = [f"{n} ({i})" for n, i in zip(df["applicant"], df["id_number"])]
        pick = st.selectbox("Applicant", labels)
        raw = df["inputs"].iloc[labels.index(pick)]
        details = json.loads(raw)
        st.dataframe(
            pd.DataFrame({"Field": list(details), "Value": [str(v) for v in details.values()]}),
            hide_index=True,
        )

    _delete_panel(everyone, "review")


# ======================================================================
# PAGE 3 - REPORTING (approvals and declines)
# ======================================================================

def _bar(labels: list[str], values: list[int], colors: list[str], title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=values,
            textposition="outside",
        )
    )
    fig.update_layout(
        font=dict(family=f"{_PAGE_FONTS['Reporting'][0]}, sans-serif"),
        title=title,
        height=320,
        yaxis=dict(title="Applicants", rangemode="tozero"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def page_reporting(art: Artifacts, threshold: float) -> None:
    heading("chart-line", "Reporting")

    _delete_notice()

    df = store.fetch_all()

    if df.empty:
        st.info("No assessments yet. Reports appear once applicants have been assessed.")
        return

    rec_approve = int((df["model_decision"] == "Approve").sum())
    rec_decline = int((df["model_decision"] == "Decline").sum())
    approved = int((df["final_decision"] == "Approved").sum())
    declined = int((df["final_decision"] == "Declined").sum())
    pending = int((df["final_decision"] == "Pending").sum())

    st.markdown("**Model recommendations**")
    m = st.columns(4)
    m[0].metric("Assessed", len(df))
    m[1].metric("Recommended approve", rec_approve)
    m[2].metric("Recommended decline", rec_decline)
    m[3].metric("Approval rate", f"{rec_approve / len(df):.0%}")

    st.markdown("**Manual review outcome**")
    r = st.columns(3)
    r[0].metric("Approved", approved)
    r[1].metric("Declined", declined)
    r[2].metric("Awaiting review", pending)

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            _bar(["Approve", "Decline"], [rec_approve, rec_decline],
                 ["#22c55e", "#ef4444"], "Model recommendations")
        )

    with right:
        st.plotly_chart(
            _bar(["Approved", "Declined", "Awaiting review"], [approved, declined, pending],
                 ["#22c55e", "#ef4444", "#f59e0b"], "Manual review outcome")
        )

    heading("clipboard-list", "All applications")

    log = pd.DataFrame(
        {
            "ID number": df["id_number"],
            "Full name": df["applicant"],
            "Submitted": df["submitted_at"],
            "Score (%)": (df["score"] * 100).round(1),
            "Model recommendation": df["model_decision"],
            "Final decision": df["final_decision"],
            "Admin note": df["admin_note"],
        }
    )

    st.dataframe(log, hide_index=True)

    st.download_button(
        "Download report (CSV)",
        log.to_csv(index=False),
        "loan_report.csv",
        "text/csv",
        key="dl_report",
    )

    _delete_panel(df, "reporting")


# ======================================================================
# PAGE 4 - AI ASSISTANT
# ======================================================================

def _api_key() -> str | None:
    """Read the OpenRouter key from Streamlit secrets or environment."""
    key = assistant.get_openrouter_api_key()
    return key or None


def _status_line(live: bool) -> str:
    def dot(color: str) -> str:
        return (
            f"<i class='fa-solid fa-circle' "
            f"style='color:{color};font-size:.55rem;"
            f"vertical-align:middle'></i>&nbsp; "
        )

    if live:
        return (
            dot("#22c55e")
            + "Powered by OpenRouter AI - answers use your current "
              "assessment and configuration."
        )

    return (
        dot("#94a3b8")
        + "Basic mode - install `openai` and set OPENROUTER_API_KEY in "
          ".streamlit/secrets.toml for free-form AI answers."
    )


def _clear_chat_button(hist: list) -> None:
    """Shown under the conversation, including right after a new answer."""
    if hist and st.button("Clear chat", key="clear_chat", help="Delete this conversation"):
        st.session_state["chat"] = []
        st.session_state.pop("pending", None)
        st.rerun()


def page_assistant(art: Artifacts, threshold: float) -> None:
    heading("robot", "AI assistant")

    key = _api_key()
    live = assistant.llm_available(key)

    st.caption(_status_line(live), unsafe_allow_html=True)

    hist = st.session_state.setdefault("chat", [])

    cols = st.columns(len(assistant.SUGGESTIONS))

    for col, suggestion in zip(cols, assistant.SUGGESTIONS):
        if col.button(suggestion, key=f"sug_{suggestion}"):
            st.session_state["pending"] = suggestion

    if not hist:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! I'm your loan assistant. Assess an applicant, "
                "then ask me why the model made its "
                "recommendation, what the score means, or how the threshold works."
            )

    for message in hist:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed = st.chat_input("Ask about the decision, the score or the model...")
    prompt = st.session_state.pop("pending", None) or typed

    if not prompt:
        _clear_chat_button(hist)
        return

    hist.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    current = st.session_state.get("assessment")

    ctx = {
        # personal details (name, ID number) are not shared with the AI
        "assessment": (
            {k: v for k, v in current.items() if k not in ("id_number", "full_name")}
            if current else None
        ),
        "applicants_table": None,
        "threshold": threshold,
        "model": art.model_label,
    }

    with st.chat_message("assistant"):
        reply = None

        if live:
            try:
                reply = st.write_stream(
                    assistant.stream_llm(key, hist, ctx)
                )
            except Exception as e:
                st.warning(
                    f"OpenRouter request failed ({type(e).__name__}). "
                    "Using the built-in assistant instead. Check your API "
                    "key, model name, account credits, and model availability."
                )

        if reply is None:
            reply = assistant.rule_based_reply(prompt, ctx)
            st.markdown(reply)

    hist.append({"role": "assistant", "content": reply})
    _clear_chat_button(hist)


# ======================================================================
# MAIN
# ======================================================================

PAGES = {
    "Assess applicant": page_assess,
    "Loan review": page_review,
    "Reporting": page_reporting,
    "AI assistant": page_assistant,
}


@st.cache_resource(show_spinner="Preparing model...")
def get_artifacts(model_path: str, model_mtime: float, code_mtime: float, config_mtime: float) -> Artifacts:
    """Cache artifacts and refresh when the model file, model.py or config.py changes."""
    return load_artifacts(Path(model_path))


def main() -> None:
    inject()
    hero()

    if not hasattr(ChoiceField, "help"):
        import model as loanmodel
        st.error(
            f"An outdated form module was loaded from `{loanmodel.__file__}`. "
            "Replace it with the latest `model.py` and restart Streamlit."
        )
        st.stop()

    if not hasattr(store, "delete"):
        st.error(
            f"An outdated database module was loaded from `{store.__file__}`. "
            "Replace it with the latest `loandb.py` and restart Streamlit."
        )
        st.stop()

    if not config.MODEL_PATH.exists():
        st.error(f"Could not find the model at `{config.MODEL_PATH}`.")
        st.caption("Expected: models/ next to the app folder, or set MODEL_PATH.")
        st.stop()

    try:
        model_code = Path(__file__).with_name("model.py")
        art = get_artifacts(
            str(config.MODEL_PATH),
            config.MODEL_PATH.stat().st_mtime,
            model_code.stat().st_mtime,
            Path(config.__file__).stat().st_mtime,
        )
    except Exception as e:
        st.error(f"Could not load the model: {e!r}")
        st.caption(
            "Use the same scikit-learn / imbalanced-learn / xgboost / "
            "lightgbm versions as the notebook (see requirements.txt)."
        )
        st.stop()

    page, threshold = render_sidebar(art, list(PAGES))
    apply_page_font(page)
    PAGES[page](art, threshold)
    footer()


main()