"""
src/preprocessing.py
Phase 1 — Preprocessing NLP batch du dataset Sentiment140 avec PySpark.

Pipeline :
    CSV brut → nettoyage textuel → tokenisation → suppression stopwords
    → HashingTF → IDF → Parquet nettoyé

Usage :
    python src/preprocessing.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pyspark.sql import functions as F
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF
)
from pyspark.ml import Pipeline

from config.config import (
    SENTIMENT140_CSV, SENTIMENT140_CLEAN_PARQUET,
    S140_COLUMNS, S140_TEXT_COL, S140_TARGET_COL, TFIDF_NUM_FEATURES
)
from src.utils import get_spark_session, clean_text_udf, ensure_dirs, get_logger

logger = get_logger(__name__)


def load_sentiment140(spark, path: str, sample_fraction: float = None):
    """
    Charge le CSV Sentiment140 (encoding latin-1).

    Args:
        spark: SparkSession
        path: Chemin vers sentiment140.csv
        sample_fraction: Si spécifié, échantillonne un sous-ensemble (tests)

    Returns:
        DataFrame Spark avec colonnes ['text', 'label']
    """
    logger.info(f"Chargement Sentiment140 depuis : {path}")

    df = (
        spark.read
        .option("header", "false")
        .option("encoding", "latin-1")
        .option("multiLine", "false")
        .option("quote", '"')
        .option("escape", '"')
        .csv(str(path))
        .toDF(*S140_COLUMNS)
    )

    # Sélection des 2 colonnes utiles pour le ML
    df = df.select(S140_TEXT_COL, S140_TARGET_COL)

    # Encodage de la variable cible : 0 (négatif) → 0, 4 (positif) → 1
    df = df.withColumn(
        "label",
        F.when(F.col(S140_TARGET_COL) == "4", 1).otherwise(0).cast("integer")
    ).drop(S140_TARGET_COL)

    if sample_fraction:
        df = df.sample(fraction=sample_fraction, seed=42)
        logger.info(f"Échantillon : {sample_fraction * 100:.0f}% du dataset")

    total = df.count()
    logger.info(f"Sentiment140 chargé : {total:,} tweets")
    return df


def clean_dataframe(df):
    """
    Applique le nettoyage textuel sur la colonne 'text'.
    Supprime les tweets vides après nettoyage.
    """
    logger.info("Nettoyage textuel en cours...")
    df = df.withColumn("text_clean", clean_text_udf(F.col("text")))
    df = df.filter(F.length(F.col("text_clean")) > 3)
    df = df.drop("text").withColumnRenamed("text_clean", "text")
    logger.info("Nettoyage terminé")
    return df


def build_nlp_features_pipeline():
    """
    Construit le pipeline Spark ML pour le feature engineering NLP.
    Ce pipeline est intégré au modèle ML complet (Phase 3).

    Étapes :
        Tokenizer → StopWordsRemover → HashingTF → IDF

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

    return Pipeline(stages=[tokenizer, sw_remover, hashing_tf, idf])


def run_preprocessing(sample_fraction: float = None):
    """
    Pipeline complet de preprocessing : CSV → Parquet nettoyé.
    """
    spark = get_spark_session("SentimentAnalysis_Preprocessing")
    ensure_dirs(SENTIMENT140_CLEAN_PARQUET.parent)

    try:
        # Chargement
        df = load_sentiment140(spark, SENTIMENT140_CSV, sample_fraction)

        # Nettoyage textuel
        df = clean_dataframe(df)

        # Statistiques de distribution des classes
        logger.info("Distribution des classes :")
        df.groupBy("label").count().show()

        # Sauvegarde en Parquet
        logger.info(f"Sauvegarde Parquet → {SENTIMENT140_CLEAN_PARQUET}")
        df.write.mode("overwrite").parquet(str(SENTIMENT140_CLEAN_PARQUET))
        logger.info("Preprocessing terminé avec succès")

        return df

    finally:
        spark.stop()
        logger.info("SparkSession fermée")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocessing Sentiment140")
    parser.add_argument("--sample", type=float, default=None,
                        help="Fraction d'échantillonnage (ex: 0.1 pour 10%)")
    args = parser.parse_args()
    run_preprocessing(sample_fraction=args.sample)