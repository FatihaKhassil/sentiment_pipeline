"""
src/evaluate_model.py
Phase 1 — Évaluation approfondie du modèle pré-entraîné.

Produit :
    - Matrice de confusion
    - Rapport de classification
    - Courbe ROC (si matplotlib disponible)
    - Analyse des erreurs (tweets mal classifiés)
    - Sauvegarde des métriques en Parquet

Usage :
    python src/evaluate_model.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)

from config.config import (
    SENTIMENT140_CSV, SENTIMENT140_CLEAN_PARQUET,
    MODEL_PATH, METRICS_DIR, SENTIMENT_METRICS_PATH,
    TEST_RATIO, RANDOM_SEED, CONFIDENCE_THRESHOLD
)
from src.preprocessing import load_sentiment140, clean_dataframe
from src.utils import (
    get_spark_session, ensure_dirs, get_logger,
    parquet_exists, get_neutral_udf, compute_reputation_score
)

logger = get_logger(__name__)


def load_trained_model(spark):
    """Charge le modèle pré-entraîné depuis le Data Lake."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle non trouvé : {MODEL_PATH}\n"
            "Lancez d'abord : python src/train_model.py"
        )
    logger.info(f"Chargement du modèle depuis : {MODEL_PATH}")
    return PipelineModel.load(str(MODEL_PATH))


def compute_confusion_matrix(predictions_df):
    """
    Calcule et affiche la matrice de confusion.

    Returns:
        dict {tp, tn, fp, fn}
    """
    cm = (
        predictions_df
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
    )
    cm.show(title="Matrice de Confusion")

    # Extraction des valeurs
    cm_dict = {row["label"] * 10 + int(row["prediction"]): row["count"]
               for row in cm.collect()}

    tp = cm_dict.get(11, 0)   # label=1, pred=1
    tn = cm_dict.get(0, 0)    # label=0, pred=0
    fp = cm_dict.get(1, 0)    # label=0, pred=1
    fn = cm_dict.get(10, 0)   # label=1, pred=0

    logger.info(f"TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def analyze_neutral_layer(predictions_df):
    """
    Analyse la distribution des labels après application de la couche Neutral.
    """
    neutral_udf = get_neutral_udf(CONFIDENCE_THRESHOLD)

    enriched = predictions_df.withColumn(
        "sentiment_label",
        neutral_udf(
            F.col("prediction"),
            F.col("probability").getItem(0),
            F.col("probability").getItem(1)
        )
    )

    logger.info("\n── Distribution avec couche métier Neutral ──")
    dist = enriched.groupBy("sentiment_label").count()
    dist_rows = {r["sentiment_label"]: r["count"] for r in dist.collect()}
    total = sum(dist_rows.values())

    for label, cnt in sorted(dist_rows.items()):
        pct = cnt / total * 100
        logger.info(f"  {label:<10} : {cnt:>8,}  ({pct:.1f}%)")

    return enriched, dist_rows


def analyze_error_samples(predictions_df, n: int = 5):
    """Affiche des exemples de tweets mal classifiés."""
    logger.info(f"\n── Exemples de tweets mal classifiés (n={n}) ──")
    errors = (
        predictions_df
        .filter(F.col("label") != F.col("prediction"))
        .select("text", "label", "prediction",
                F.col("probability").getItem(0).alias("p_neg"),
                F.col("probability").getItem(1).alias("p_pos"))
        .limit(n)
    )
    errors.show(truncate=60)
    return errors


def save_metrics_parquet(spark, metrics: dict, cm: dict):
    """Sauvegarde les métriques en Parquet pour le dashboard."""
    ensure_dirs(METRICS_DIR)

    data = [{
        "metric_name": k,
        "metric_value": float(v),
        "evaluation_set": "test_sentiment140"
    } for k, v in {**metrics, **cm}.items()]

    df = spark.createDataFrame(data)
    df.write.mode("overwrite").parquet(str(SENTIMENT_METRICS_PATH))
    logger.info(f"Métriques sauvegardées → {SENTIMENT_METRICS_PATH}")


def run_evaluation():
    """Pipeline complet d'évaluation."""
    spark = get_spark_session("SentimentAnalysis_Evaluation")
    ensure_dirs(METRICS_DIR)

    try:
        # ── Chargement du modèle ──────────────────────────────────────────────
        model = load_trained_model(spark)

        # ── Chargement des données ────────────────────────────────────────────
        if parquet_exists(SENTIMENT140_CLEAN_PARQUET):
            df = spark.read.parquet(str(SENTIMENT140_CLEAN_PARQUET))
        else:
            df = load_sentiment140(spark, SENTIMENT140_CSV)
            df = clean_dataframe(df)

        # ── Jeu de test (même split que l'entraînement) ───────────────────────
        _, test_df = df.randomSplit([1 - TEST_RATIO, TEST_RATIO], seed=RANDOM_SEED)
        logger.info(f"Jeu de test : {test_df.count():,} tweets")

        # ── Prédictions ───────────────────────────────────────────────────────
        predictions = model.transform(test_df)
        predictions.cache()

        # ── Métriques ─────────────────────────────────────────────────────────
        from src.train_model import evaluate_model
        metrics = evaluate_model(predictions)

        # ── Matrice de confusion ──────────────────────────────────────────────
        cm = compute_confusion_matrix(predictions)

        # ── Analyse couche Neutral ────────────────────────────────────────────
        enriched, dist = analyze_neutral_layer(predictions)

        # ── Exemples d'erreurs ────────────────────────────────────────────────
        analyze_error_samples(predictions)

        # ── Sauvegarde métriques ──────────────────────────────────────────────
        save_metrics_parquet(spark, metrics, cm)

        return metrics

    finally:
        spark.stop()
        logger.info("SparkSession fermée")


if __name__ == "__main__":
    run_evaluation()