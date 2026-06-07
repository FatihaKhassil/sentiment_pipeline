"""
dashboard/app.py
Dashboard Streamlit — Real-Time Sentiment Analysis & Apple Brand Monitoring

Fonctionnalités :
    - KPIs temps réel (Positifs / Négatifs / Neutres / Score Réputation)
    - Évolution temporelle des sentiments (courbe empilée)
    - Distribution par produit Apple
    - Jauge de réputation
    - Détection de Bad Buzz avec alertes visuelles
    - Top 10 tweets les plus négatifs
    - Filtre par source : Sentiment140 ou Apple

Usage :
    streamlit run dashboard/app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.config import (
    SENTIMENT140_PREDICTIONS_DIR, APPLE_PREDICTIONS_DIR,
    BAD_BUZZ_THRESHOLD, BAD_BUZZ_WINDOW_MINUTES,
    CONFIDENCE_THRESHOLD, DASHBOARD_REFRESH_SECONDS,
    TOP_NEGATIVE_TWEETS_N, DASHBOARD_TITLE
)
from src.utils import compute_reputation_score, get_bad_buzz_status

# ─── Palette (thème sombre, inspiré du mockup fourni) ─────────────────────────
COLOR_BG        = "#0E1117"   # Fond général
COLOR_CARD      = "#171B23"   # Fond des cartes
COLOR_CARD_ALT  = "#1B2028"   # Fond cartes secondaires / sidebar
COLOR_BORDER    = "#2A2F3A"   # Bordures discrètes
COLOR_ACCENT    = "#3B82F6"   # Bleu accent
COLOR_POSITIVE  = "#22C55E"   # Vert
COLOR_NEGATIVE  = "#EF4444"   # Rouge
COLOR_NEUTRAL   = "#9CA3AF"   # Gris
COLOR_TEXT      = "#F5F6F8"   # Texte principal (blanc cassé — forte lisibilité sur fond sombre)
COLOR_SUBTEXT   = "#C7CCD4"   # Texte secondaire (gris clair — reste lisible sur fond sombre)

SENTIMENT_COLORS = {"Positive": COLOR_POSITIVE, "Negative": COLOR_NEGATIVE, "Neutral": COLOR_NEUTRAL}

# ─── Configuration Streamlit ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS personnalisé — thème sombre, cartes arrondies ────────────────────────
st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }}
    .stApp {{
        background-color: {COLOR_BG}; color: {COLOR_TEXT};
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{
        background-color: {COLOR_BG} !important; background-image: none !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {COLOR_CARD_ALT}; border-right: 1px solid {COLOR_BORDER};
    }}
    [data-testid="stSidebar"] * {{
        color: {COLOR_TEXT};
    }}
    [data-testid="stCaptionContainer"], small, .stCaption {{
        color: {COLOR_SUBTEXT} !important;
    }}
    .header-title {{
        font-size: 1.9rem; font-weight: 600; color: {COLOR_TEXT};
        margin-bottom: 0.1rem; letter-spacing: -0.01rem;
    }}
    .header-subtitle {{
        font-size: 0.9rem; color: {COLOR_SUBTEXT}; margin-bottom: 1rem;
    }}
    .section-title {{
        font-size: 1.05rem; font-weight: 600; color: {COLOR_TEXT};
        margin: 0.25rem 0 0.75rem 0; border-left: 3px solid {COLOR_ACCENT};
        padding-left: 0.6rem;
    }}
    .status-banner {{
        border-radius: 10px; padding: 0.9rem 1.2rem; font-size: 0.95rem;
        font-weight: 600; border: 1px solid; margin-bottom: 0.5rem;
    }}
    .status-alert {{
        background: rgba(239,68,68,0.12); border-color: {COLOR_NEGATIVE}; color: #FCA5A5;
    }}
    .status-watch {{
        background: rgba(245,158,11,0.12); border-color: #F59E0B; color: #FCD34D;
    }}
    .status-stable {{
        background: rgba(34,197,94,0.12); border-color: {COLOR_POSITIVE}; color: #86EFAC;
    }}
    .status-nodata {{
        background: {COLOR_CARD}; border-color: {COLOR_BORDER}; color: {COLOR_SUBTEXT};
    }}
    /* Cartes KPI */
    [data-testid="stMetric"] {{
        background: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 14px;
        padding: 1rem 1.1rem 0.8rem 1.1rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLOR_SUBTEXT}; font-size: 0.82rem; font-weight: 500;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLOR_TEXT}; font-weight: 700;
    }}
    /* Conteneurs de graphiques */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px;
    }}
    div[data-testid="stExpander"] {{
        background: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 14px;
    }}
    [data-testid="stDataFrame"] {{
        border-radius: 10px; overflow: hidden;
    }}
    hr {{
        border-color: {COLOR_BORDER};
    }}
</style>
""", unsafe_allow_html=True)


# ─── Chargement des données ────────────────────────────────────────────────────

@st.cache_data(ttl=DASHBOARD_REFRESH_SECONDS)
def load_predictions(parquet_dir: Path, max_rows: int = 100_000) -> pd.DataFrame:
    """
    Charge les prédictions depuis le Data Lake Parquet.
    Retourne un DataFrame vide si aucune donnée n'est disponible.
    """
    if not parquet_dir.exists():
        return pd.DataFrame()

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame()

    dfs = []
    for pf in parquet_files[-50:]:   # Derniers 50 fichiers max
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)

    # Conversion du timestamp
    for col in ["processing_time", "kafka_timestamp"]:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], errors="coerce")

    # Tri par temps de traitement
    if "processing_time" in result.columns:
        result = result.sort_values("processing_time", ascending=False)

    return result.head(max_rows)


def compute_kpis(df: pd.DataFrame) -> dict:
    """Calcule les KPIs principaux à partir du DataFrame de prédictions."""
    if df.empty:
        return {
            "total": 0, "positive": 0, "negative": 0, "neutral": 0,
            "pct_pos": 0.0, "pct_neg": 0.0, "pct_neutral": 0.0,
            "reputation_score": 50.0, "status": "NO_DATA"
        }

    total = len(df)
    n_pos = (df["sentiment_label"] == "Positive").sum()
    n_neg = (df["sentiment_label"] == "Negative").sum()
    n_neu = (df["sentiment_label"] == "Neutral").sum()

    return {
        "total":          total,
        "positive":       int(n_pos),
        "negative":       int(n_neg),
        "neutral":        int(n_neu),
        "pct_pos":        round(n_pos / total * 100, 1),
        "pct_neg":        round(n_neg / total * 100, 1),
        "pct_neutral":    round(n_neu / total * 100, 1),
        "reputation_score": compute_reputation_score(int(n_pos), int(n_neg)),
        "status":         get_bad_buzz_status(int(n_neg), int(n_pos), BAD_BUZZ_THRESHOLD)
    }


# ─── Composants visuels ────────────────────────────────────────────────────────

def render_kpi_row(kpis: dict):
    """Affiche la rangée de KPIs principaux."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Tweets", f"{kpis['total']:,}")
    with col2:
        st.metric("Positive", f"{kpis['pct_pos']:.1f}%",
                  delta=f"{kpis['positive']:,} tweets")
    with col3:
        st.metric("Negative", f"{kpis['pct_neg']:.1f}%",
                  delta=f"{kpis['negative']:,} tweets", delta_color="inverse")
    with col4:
        st.metric("Neutral", f"{kpis['pct_neutral']:.1f}%",
                  delta=f"{kpis['neutral']:,} tweets")
    with col5:
        st.metric("Reputation Score", f"{kpis['reputation_score']:.1f} / 100")


def render_bad_buzz_alert(kpis: dict):
    """Affiche le statut de la marque (alerte, surveillance ou stable)."""
    status = kpis["status"]
    score = kpis["reputation_score"]

    if status == "BAD_BUZZ":
        st.markdown(
            f"<div class='status-banner status-alert'>Alert — Negative sentiment threshold exceeded "
            f"&nbsp;|&nbsp; Reputation score: {score:.1f}/100 "
            f"&nbsp;|&nbsp; Negative rate: {kpis['pct_neg']:.1f}% "
            f"(threshold: {BAD_BUZZ_THRESHOLD*100:.0f}%)</div>",
            unsafe_allow_html=True
        )
    elif status == "SURVEILLANCE":
        st.markdown(
            f"<div class='status-banner status-watch'>Monitoring — Elevated negative sentiment "
            f"&nbsp;|&nbsp; Reputation score: {score:.1f}/100 "
            f"&nbsp;|&nbsp; Negative rate: {kpis['pct_neg']:.1f}%</div>",
            unsafe_allow_html=True
        )
    elif status == "NO_DATA":
        st.markdown(
            "<div class='status-banner status-nodata'>Awaiting data — start the Kafka producer "
            "to begin streaming tweets.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='status-banner status-stable'>Stable — "
            f"Reputation score: {score:.1f}/100</div>",
            unsafe_allow_html=True
        )


def render_sentiment_timeline(df: pd.DataFrame):
    """Courbe d'évolution temporelle des sentiments (fenêtre glissante 5 min)."""
    if df.empty or "processing_time" not in df.columns:
        st.caption("No time-series data available yet.")
        return

    df_time = df.dropna(subset=["processing_time"]).copy()
    df_time["minute"] = df_time["processing_time"].dt.floor("T")

    timeline = (
        df_time.groupby(["minute", "sentiment_label"])
        .size()
        .reset_index(name="count")
        .pivot(index="minute", columns="sentiment_label", values="count")
        .fillna(0)
        .reset_index()
    )

    # S'assurer que les 3 colonnes existent
    for col in ["Positive", "Negative", "Neutral"]:
        if col not in timeline.columns:
            timeline[col] = 0

    fig = go.Figure()

    for label in ["Positive", "Negative", "Neutral"]:
        fig.add_trace(go.Scatter(
            x=timeline["minute"], y=timeline[label],
            name=label, mode="lines",
            line=dict(color=SENTIMENT_COLORS[label], width=2),
        ))

    fig.update_layout(
        title="Sentiment trend over time (per minute)",
        xaxis_title="Time", yaxis_title="Number of tweets",
        height=370,
        font=dict(family="Segoe UI, sans-serif", color=COLOR_TEXT),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=60, r=20, b=60),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.28, x=0, title=None)
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLOR_BORDER, title_standoff=15, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_BORDER, title_standoff=15, automargin=True)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_sentiment_pie(kpis: dict):
    """Camembert Positive / Negative / Neutral."""
    labels = ["Positive", "Negative", "Neutral"]
    values = [kpis["positive"], kpis["negative"], kpis["neutral"]]
    colors = [SENTIMENT_COLORS[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colors, hole=0.55,
        textinfo="label+percent"
    ))
    fig.update_layout(
        title="Sentiment distribution",
        height=300,
        showlegend=False,
        font=dict(family="Segoe UI, sans-serif", color=COLOR_TEXT),
        margin=dict(t=50, l=10, r=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_product_distribution(df: pd.DataFrame):
    """Distribution des sentiments par produit Apple (histogramme groupé)."""
    if "product" not in df.columns or df.empty:
        st.caption("Product-level data is not available.")
        return

    prod_df = (
        df.groupby(["product", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        prod_df, x="product", y="count", color="sentiment_label",
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment by product"
    )
    fig.update_layout(
        height=320, xaxis_title="Product", yaxis_title="Number of tweets",
        legend_title="Sentiment",
        font=dict(family="Segoe UI, sans-serif", color=COLOR_TEXT),
        margin=dict(t=50, l=60, r=20, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(showgrid=False, title_standoff=15, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_BORDER, title_standoff=15, automargin=True)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_top_negative_tweets(df: pd.DataFrame, n: int = TOP_NEGATIVE_TWEETS_N):
    """Tableau des tweets les plus négatifs (prob_negative > 0.85)."""
    if df.empty or "prob_negative" not in df.columns:
        return

    top_neg = (
        df[df["prob_negative"] > 0.75]
        .nlargest(n, "prob_negative")
        [["text", "prob_negative", "product"] if "product" in df.columns else ["text", "prob_negative"]]
    )

    if top_neg.empty:
        st.caption("No strongly negative tweets (threshold 75%) detected at the moment.")
        return

    top_neg = top_neg.rename(columns={
        "text": "Tweet text",
        "prob_negative": "P(negative)",
        "product": "Product"
    })
    top_neg["P(negative)"] = top_neg["P(negative)"].round(3)
    st.dataframe(top_neg, use_container_width=True, hide_index=True)


def render_confidence_histogram(df: pd.DataFrame):
    """Histogramme de distribution des scores de confiance."""
    if df.empty or "confidence" not in df.columns:
        return

    fig = px.histogram(
        df, x="confidence", color="sentiment_label", nbins=50,
        color_discrete_map=SENTIMENT_COLORS,
        title="Model confidence distribution"
    )
    fig.add_vline(x=CONFIDENCE_THRESHOLD, line_dash="dash",
                  line_color=COLOR_ACCENT,
                  annotation_text=f"Neutral threshold ({CONFIDENCE_THRESHOLD})")
    fig.update_layout(height=320, legend_title="Sentiment",
                      xaxis_title="Confidence", yaxis_title="Number of tweets",
                      font=dict(family="Segoe UI, sans-serif", color=COLOR_TEXT),
                      margin=dict(t=50, l=60, r=20, b=60),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, title_standoff=15, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_BORDER, title_standoff=15, automargin=True)
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ─── Layout principal ─────────────────────────────────────────────────────────

def main():
    # ── En-tête ───────────────────────────────────────────────────────────────
    st.markdown(f"<div class='header-title'>{DASHBOARD_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='header-subtitle'>Monitor brand reputation and detect sentiment shifts as they happen.</div>",
        unsafe_allow_html=True
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("**Settings**")

        source = st.radio(
            "Data source",
            ["Apple Brand Monitoring", "General Public Stream"]
        )

        st.divider()
        auto_refresh = st.checkbox("Auto-refresh", value=True)

        st.divider()
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()

    # ── Chargement des données ────────────────────────────────────────────────
    is_apple = "Apple" in source
    data_path = APPLE_PREDICTIONS_DIR if is_apple else SENTIMENT140_PREDICTIONS_DIR

    with st.spinner("Loading predictions from the data lake..."):
        df = load_predictions(data_path)

    # ── Filtre produit Apple ──────────────────────────────────────────────────
    if is_apple and not df.empty and "product" in df.columns:
        products = ["All"] + sorted(df["product"].dropna().unique().tolist())
        selected_product = st.sidebar.selectbox("Filter by product", products)
        if selected_product != "All":
            df = df[df["product"] == selected_product]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis = compute_kpis(df)

    st.markdown(
        f"<div class='section-title'>"
        f"{'Apple Brand Monitoring' if is_apple else 'General Public Stream'}</div>",
        unsafe_allow_html=True
    )
    render_bad_buzz_alert(kpis)
    st.markdown("<br>", unsafe_allow_html=True)
    render_kpi_row(kpis)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphiques — regroupés en haut, dans des cadres délimités ─────────────
    with st.container(border=True):
        col_trend, col_dist = st.columns([2, 1])
        with col_trend:
            render_sentiment_timeline(df)
        with col_dist:
            render_sentiment_pie(kpis)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        if is_apple:
            col_a, col_b = st.columns(2)
            with col_a:
                render_confidence_histogram(df)
            with col_b:
                render_product_distribution(df)
        else:
            render_confidence_histogram(df)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sections textuelles — regroupées en bas ──────────────────────────────
    with st.container(border=True):
        st.markdown(f"<div class='section-title'>Top {TOP_NEGATIVE_TWEETS_N} most negative tweets</div>",
                    unsafe_allow_html=True)
        render_top_negative_tweets(df)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Raw data (most recent processed tweets)"):
        if not df.empty:
            cols_to_show = [c for c in
                ["text", "sentiment_label", "confidence", "product", "processing_time"]
                if c in df.columns]
            st.dataframe(df[cols_to_show].head(50), use_container_width=True, hide_index=True)
        else:
            st.warning("No data available. Is the streaming pipeline running?")

    # ── Rafraîchissement automatique ──────────────────────────────────────────
    if auto_refresh:
        time.sleep(DASHBOARD_REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()