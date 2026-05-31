"""
src/train_model.py
Phase 1 — Entraînement du modèle Logistic Regression avec Spark MLlib.

Pipeline complet :
    Parquet nettoyé → Pipeline NLP (TF-IDF) + Logistic Regression
    → Évaluation → Sauvegarde modèle

Usage :
    python src/train_model.py
    python src/train_model.py --sample 0.1   # test rapide sur 10%
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

from config.config import (
    SENTIMENT140_CSV, SENTIMENT140_CLEAN_PARQUET, MODEL_PATH,
    TRAIN_RATIO, TEST_RATIO, RANDOM_SEED,
    LR_MAX_ITER, LR_REG_PARAM, TFIDF_NUM_FEATURES
)
from src.preprocessing import load_sentiment140, clean_dataframe
from src.utils import get_spark_session, ensure_dirs, get_logger, parquet_exists

logger = get_logger(__name__)


def build_full_ml_pipeline():
    """
    Construit le pipeline ML complet :
        NLP Features (TF-IDF) + Logistic Regression

    Ce pipeline encapsule TOUT le traitement nécessaire à l'inférence :
    seul le texte brut est nécessaire à l'entrée.

    Returns:
        Pipeline Spark ML non ajusté
    """
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens_raw")

    sw_remover = StopWordsRemover(
        inputCol="tokens_raw",
        outputCol="tokens",
        caseSensitive=False
    )

    hashing_tf = HashingTF(
        inputCol="tokens",
        outputCol="raw_features",
        numFeatures=TFIDF_NUM_FEATURES
    )

    idf = IDF(
        inputCol="raw_features",
        outputCol="features",
        minDocFreq=2
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=LR_MAX_ITER,
        regParam=LR_REG_PARAM,
        elasticNetParam=0.0,
        family="binomial"
    )

    return Pipeline(stages=[tokenizer, sw_remover, hashing_tf, idf, lr])


def evaluate_model(predictions_df):
    """
    Calcule les métriques de performance sur le jeu de test.

    Args:
        predictions_df: DataFrame avec colonnes 'label' et 'prediction'

    Returns:
        dict avec accuracy, precision, recall, f1, auc_roc
    """
    binary_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    multi_eval_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    multi_eval_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    multi_eval_prec = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="weightedPrecision"
    )
    multi_eval_rec = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="weightedRecall"
    )

    metrics = {
        "accuracy":  round(multi_eval_acc.evaluate(predictions_df), 4),
        "precision": round(multi_eval_prec.evaluate(predictions_df), 4),
        "recall":    round(multi_eval_rec.evaluate(predictions_df), 4),
        "f1_score":  round(multi_eval_f1.evaluate(predictions_df), 4),
        "auc_roc":   round(binary_eval.evaluate(predictions_df), 4),
    }

    logger.info("=" * 50)
    logger.info("MÉTRIQUES DU MODÈLE — Jeu de test Sentiment140")
    logger.info("=" * 50)
    for name, val in metrics.items():
        target = {"accuracy": 0.78, "precision": 0.76, "recall": 0.76,
                  "f1_score": 0.77, "auc_roc": 0.85}.get(name, 0)
        status = "✓" if val >= target else "✗"
        logger.info(f"  {status}  {name:<12} : {val:.4f}  (cible ≥ {target})")
    logger.info("=" * 50)

    return metrics


def run_training(sample_fraction: float = None):
    """
    Pipeline complet d'entraînement Phase 1.
    """
    spark = get_spark_session("SentimentAnalysis_BatchTraining")
    ensure_dirs(MODEL_PATH.parent)

    try:
        # ── Chargement des données ────────────────────────────────────────────
        if parquet_exists(SENTIMENT140_CLEAN_PARQUET):
            logger.info(f"Chargement depuis Parquet nettoyé : {SENTIMENT140_CLEAN_PARQUET}")
            df = spark.read.parquet(str(SENTIMENT140_CLEAN_PARQUET))
        else:
            logger.info("Parquet non trouvé — rechargement depuis CSV brut")
            df = load_sentiment140(spark, SENTIMENT140_CSV, sample_fraction)
            df = clean_dataframe(df)

        if sample_fraction and not parquet_exists(SENTIMENT140_CLEAN_PARQUET):
            df = df.sample(fraction=sample_fraction, seed=RANDOM_SEED)

        total = df.count()
        logger.info(f"Dataset chargé : {total:,} tweets")

        # ── Split Train / Test ────────────────────────────────────────────────
        train_df, test_df = df.randomSplit(
            [TRAIN_RATIO, TEST_RATIO], seed=RANDOM_SEED
        )
        logger.info(f"Train : {train_df.count():,} | Test : {test_df.count():,}")

        # ── Construction et entraînement du pipeline ──────────────────────────
        logger.info("Entraînement du pipeline ML (TF-IDF + Logistic Regression)...")
        pipeline = build_full_ml_pipeline()
        pipeline_model = pipeline.fit(train_df)
        logger.info("Entraînement terminé")

        # ── Évaluation ────────────────────────────────────────────────────────
        logger.info("Évaluation sur le jeu de test...")
        predictions = pipeline_model.transform(test_df)
        metrics = evaluate_model(predictions)

        # ── Sauvegarde du modèle ──────────────────────────────────────────────
        logger.info(f"Sauvegarde du modèle → {MODEL_PATH}")
        pipeline_model.write().overwrite().save(str(MODEL_PATH))
        logger.info("Modèle sauvegardé avec succès")

        # Structure sauvegardée :
        # data/models/sentiment_model/
        #   ├── metadata/   (hyperparamètres, schéma)
        #   └── stages/     (tokenizer, TF, IDF, LR)

        return pipeline_model, metrics

    finally:
        spark.stop()
        logger.info("SparkSession fermée")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entraînement modèle Sentiment140")
    parser.add_argument("--sample", type=float, default=None,
                        help="Fraction d'échantillonnage pour test rapide")
    args = parser.parse_args()
    run_training(sample_fraction=args.sample)