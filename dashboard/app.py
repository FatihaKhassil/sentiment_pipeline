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
from datetime import datetime, timedelta

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

# ─── Configuration Streamlit ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
    }
    .bad-buzz-alert {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        padding: 1rem; border-radius: 10px; color: white; font-weight: bold;
        font-size: 1.1rem; text-align: center; animation: pulse 1s infinite;
    }
    .stable-status {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
    }
    .surveillance-status {
        background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
    }
    .header-title {
        font-size: 2rem; font-weight: bold;
        background: linear-gradient(90deg, #1e3a5f, #2980b9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
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
        st.metric("📊 Total Tweets", f"{kpis['total']:,}")
    with col2:
        st.metric("😊 Positifs", f"{kpis['pct_pos']:.1f}%",
                  delta=f"{kpis['positive']:,} tweets")
    with col3:
        st.metric("😤 Négatifs", f"{kpis['pct_neg']:.1f}%",
                  delta=f"{kpis['negative']:,} tweets", delta_color="inverse")
    with col4:
        st.metric("😐 Neutres", f"{kpis['pct_neutral']:.1f}%",
                  delta=f"{kpis['neutral']:,} tweets")
    with col5:
        st.metric("⭐ Score Réputation", f"{kpis['reputation_score']:.1f}/100")


def render_bad_buzz_alert(kpis: dict):
    """Affiche le statut de la marque (alerte, surveillance ou stable)."""
    status = kpis["status"]
    score = kpis["reputation_score"]

    if status == "BAD_BUZZ":
        st.markdown(
            f"<div class='bad-buzz-alert'>🚨 ALERTE BAD BUZZ — "
            f"Score Réputation : {score:.1f}/100 — "
            f"Taux négatif : {kpis['pct_neg']:.1f}% (seuil : {BAD_BUZZ_THRESHOLD*100:.0f}%)</div>",
            unsafe_allow_html=True
        )
    elif status == "SURVEILLANCE":
        st.markdown(
            f"<div class='surveillance-status'>⚠️ SURVEILLANCE — "
            f"Score : {score:.1f}/100 — "
            f"Taux négatif : {kpis['pct_neg']:.1f}%</div>",
            unsafe_allow_html=True
        )
    elif status == "NO_DATA":
        st.info("⏳ En attente de données... Lancez le producteur Kafka.")
    else:
        st.markdown(
            f"<div class='stable-status'>✅ STABLE — "
            f"Score Réputation : {score:.1f}/100</div>",
            unsafe_allow_html=True
        )


def render_sentiment_timeline(df: pd.DataFrame):
    """Courbe d'évolution temporelle des sentiments (fenêtre glissante 5 min)."""
    if df.empty or "processing_time" not in df.columns:
        st.info("Pas encore de données temporelles")
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
    colors = {"Positive": "#27ae60", "Negative": "#e74c3c", "Neutral": "#95a5a6"}

    for label in ["Positive", "Negative", "Neutral"]:
        fig.add_trace(go.Scatter(
            x=timeline["minute"], y=timeline[label],
            name=label, mode="lines+markers",
            line=dict(color=colors[label], width=2),
            fill="tozeroy", fillcolor=colors[label].replace(")", ",0.15)").replace("rgb", "rgba") if "#" not in colors[label] else None
        ))

    fig.update_layout(
        title="📈 Évolution temporelle des sentiments (par minute)",
        xaxis_title="Temps", yaxis_title="Nombre de tweets",
        legend_title="Sentiment", height=350,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment_pie(kpis: dict):
    """Camembert Positive / Negative / Neutral."""
    labels = ["Positifs", "Négatifs", "Neutres"]
    values = [kpis["positive"], kpis["negative"], kpis["neutral"]]
    colors = ["#27ae60", "#e74c3c", "#95a5a6"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colors, hole=0.4,
        textinfo="label+percent"
    ))
    fig.update_layout(
        title="🥧 Répartition des sentiments",
        height=300,
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_reputation_gauge(score: float):
    """Jauge de réputation [0, 100]."""
    color = "#e74c3c" if score < 30 else "#e67e22" if score < 50 else "#27ae60"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 50},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Score de Réputation"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#fadbd8"},
                {"range": [30, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#d5f5e3"}
            ],
            "threshold": {
                "line": {"color": "#1e3a5f", "width": 4},
                "thickness": 0.75, "value": 50
            }
        }
    ))
    fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def render_product_distribution(df: pd.DataFrame):
    """Distribution des sentiments par produit Apple (histogramme groupé)."""
    if "product" not in df.columns or df.empty:
        st.info("Données par produit Apple non disponibles")
        return

    prod_df = (
        df.groupby(["product", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        prod_df, x="product", y="count", color="sentiment_label",
        barmode="group",
        color_discrete_map={
            "Positive": "#27ae60",
            "Negative": "#e74c3c",
            "Neutral":  "#95a5a6"
        },
        title="📱 Distribution des sentiments par produit Apple"
    )
    fig.update_layout(
        height=350, xaxis_title="Produit", yaxis_title="Nombre de tweets",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


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
        st.info("Aucun tweet très négatif (seuil 75%) détecté pour le moment")
        return

    top_neg = top_neg.rename(columns={
        "text": "Texte du tweet",
        "prob_negative": "P(Négatif)",
        "product": "Produit"
    })
    top_neg["P(Négatif)"] = top_neg["P(Négatif)"].round(3)
    st.dataframe(top_neg, use_container_width=True)


def render_confidence_histogram(df: pd.DataFrame):
    """Histogramme de distribution des scores de confiance."""
    if df.empty or "confidence" not in df.columns:
        return

    fig = px.histogram(
        df, x="confidence", color="sentiment_label", nbins=50,
        color_discrete_map={
            "Positive": "#27ae60", "Negative": "#e74c3c", "Neutral": "#95a5a6"
        },
        title="📊 Distribution des scores de confiance ML"
    )
    fig.add_vline(x=CONFIDENCE_THRESHOLD, line_dash="dash",
                  line_color="#1e3a5f",
                  annotation_text=f"Seuil Neutral ({CONFIDENCE_THRESHOLD})")
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ─── Layout principal ─────────────────────────────────────────────────────────

def main():
    # ── En-tête ───────────────────────────────────────────────────────────────
    st.markdown(f"<div class='header-title'>📊 {DASHBOARD_TITLE}</div>",
                unsafe_allow_html=True)
    st.caption(f"Kafka · Spark Structured Streaming · MLlib · Parquet | "
               f"Rafraîchissement : {DASHBOARD_REFRESH_SECONDS}s")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.shields.io/badge/Kafka-231F20?style=flat&logo=apache-kafka",
                 use_column_width=True)
        st.title("🎛️ Paramètres")

        source = st.radio(
            "📂 Source de données",
            ["Apple Brand Monitoring (Phase 3)", "Sentiment140 Big Data (Phase 2)"]
        )

        st.divider()
        st.subheader("⚙️ Seuils")
        conf_threshold = st.slider(
            "Seuil Neutral (confiance ML)",
            min_value=0.50, max_value=0.90, value=CONFIDENCE_THRESHOLD, step=0.05
        )
        buzz_threshold = st.slider(
            "Seuil Bad Buzz (taux négatif)",
            min_value=0.50, max_value=0.90, value=BAD_BUZZ_THRESHOLD, step=0.05
        )

        st.divider()
        auto_refresh = st.checkbox("🔄 Rafraîchissement automatique", value=True)

        st.divider()
        st.subheader("📌 Architecture")
        st.markdown("""
        **Phase 1** — Batch Training  
        `Sentiment140 → Spark MLlib`
        
        **Phase 2** — Big Data Streaming  
        `Kafka → Spark → Parquet`
        
        **Phase 3** — Brand Monitoring  
        `Apple Tweets → Dashboard`
        """)

        st.divider()
        if st.button("🔄 Actualiser maintenant"):
            st.cache_data.clear()
            st.rerun()

    # ── Chargement des données ────────────────────────────────────────────────
    is_apple = "Apple" in source
    data_path = APPLE_PREDICTIONS_DIR if is_apple else SENTIMENT140_PREDICTIONS_DIR

    with st.spinner("Chargement des prédictions depuis le Data Lake..."):
        df = load_predictions(data_path)

    # ── Filtre produit Apple ──────────────────────────────────────────────────
    if is_apple and not df.empty and "product" in df.columns:
        products = ["Tous"] + sorted(df["product"].dropna().unique().tolist())
        selected_product = st.sidebar.selectbox("📱 Filtrer par produit", products)
        if selected_product != "Tous":
            df = df[df["product"] == selected_product]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis = compute_kpis(df)

    st.subheader(f"{'🍎 Apple Brand Monitoring' if is_apple else '📡 Sentiment140 Big Data Stream'}")
    render_bad_buzz_alert(kpis)
    st.divider()
    render_kpi_row(kpis)
    st.divider()

    # ── Visualisations ────────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        render_sentiment_timeline(df)

    with col_right:
        render_reputation_gauge(kpis["reputation_score"])

    col1, col2 = st.columns(2)

    with col1:
        render_sentiment_pie(kpis)

    with col2:
        render_confidence_histogram(df)

    # Visualisations spécifiques Apple
    if is_apple:
        render_product_distribution(df)

    # ── Top tweets négatifs ───────────────────────────────────────────────────
    st.subheader(f"🚨 Top {TOP_NEGATIVE_TWEETS_N} Tweets les plus négatifs")
    render_top_negative_tweets(df)

    # ── Données brutes ────────────────────────────────────────────────────────
    with st.expander("🗃️ Données brutes (derniers tweets traités)"):
        if not df.empty:
            cols_to_show = [c for c in
                ["text", "sentiment_label", "confidence", "product", "processing_time"]
                if c in df.columns]
            st.dataframe(df[cols_to_show].head(50), use_container_width=True)
        else:
            st.warning("Aucune donnée disponible. Le pipeline de streaming est-il actif ?")

    # ── Pied de page ──────────────────────────────────────────────────────────
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.caption(f"⏱️ Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
    with col_b:
        st.caption(f"📦 {kpis['total']:,} tweets analysés")
    with col_c:
        st.caption(f"🎯 Seuil confiance : {conf_threshold} | Seuil buzz : {buzz_threshold*100:.0f}%")

    # ── Rafraîchissement automatique ──────────────────────────────────────────
    if auto_refresh:
        time.sleep(DASHBOARD_REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()