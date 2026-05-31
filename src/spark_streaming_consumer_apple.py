"""
src/spark_streaming_consumer_apple.py
Phase 3 — Consommateur Spark Structured Streaming pour Apple Brand Monitoring.

Pipeline identique à Phase 2 avec adaptation pour les tweets Apple :
    Kafka Topic 'apple_tweets'
    → JSON parsing (tweet_text → renommé 'text' pour le modèle)
    → Inférence ML (même modèle Phase 1)
    → Couche Métier Neutral
    → Parquet Data Lake (apple_predictions/)
    → Agrégations par produit (fenêtre glissante)

Usage :
    python src/spark_streaming_consumer_apple.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType
)
from pyspark.ml import PipelineModel

from config.config import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_APPLE,
    MODEL_PATH, APPLE_PREDICTIONS_DIR, CHECKPOINT_DIR_APPLE,
    STREAMING_TRIGGER_SECONDS, STREAMING_MAX_OFFSETS,
    CONFIDENCE_THRESHOLD, BAD_BUZZ_THRESHOLD, BAD_BUZZ_WINDOW_MINUTES
)
from src.utils import (
    get_spark_kafka_session, ensure_dirs, get_logger,
    clean_text_udf, get_neutral_udf
)

logger = get_logger(__name__)


# ── Schéma du message Kafka Apple Tweets ─────────────────────────────────────
APPLE_MESSAGE_SCHEMA = StructType([
    StructField("tweet_text", StringType(), True),
    StructField("product", StringType(), True),
    StructField("annotation", StringType(), True),
    StructField("ingestion_timestamp", StringType(), True)
])


def create_apple_kafka_stream(spark, bootstrap_servers: str, topic: str):
    """Crée le flux Kafka pour le topic Apple Tweets."""
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", STREAMING_MAX_OFFSETS)
        .option("failOnDataLoss", "false")
        .load()
    )


def parse_apple_messages(raw_stream):
    """
    Parse les messages Kafka Apple.
    Renomme 'tweet_text' → 'text' pour compatibilité avec le modèle ML.
    """
    return (
        raw_stream
        .select(
            F.from_json(
                F.col("value").cast("string"),
                APPLE_MESSAGE_SCHEMA
            ).alias("data"),
            F.col("timestamp").alias("kafka_timestamp")
        )
        .select("data.*", "kafka_timestamp")
        # Renommage clé : tweet_text → text (attendu par le pipeline ML)
        .withColumnRenamed("tweet_text", "text")
        .filter(F.col("text").isNotNull() & (F.length(F.col("text")) > 0))
        .filter(~F.lower(F.col("text")).isin("nan", "null", "none", ""))
    )


def apply_ml_and_neutral(df, pipeline_model: PipelineModel):
    """
    Applique le pipeline ML + couche Neutral sur les tweets Apple.
    Identique à Phase 2 — démontre la réutilisabilité du pipeline.
    """
    neutral_udf = get_neutral_udf(CONFIDENCE_THRESHOLD)

    # Nettoyage textuel
    df = df.withColumn("text", clean_text_udf(F.col("text")))
    df = df.filter(F.length(F.col("text")) > 2)

    # Inférence ML (même modèle que Phase 2)
    predictions = pipeline_model.transform(df)

    # Couche Métier Neutral
    enriched = (
        predictions
        .withColumn(
            "sentiment_label",
            neutral_udf(
                F.col("prediction"),
                F.col("probability").getItem(0),
                F.col("probability").getItem(1)
            )
        )
        .withColumn("prob_negative", F.col("probability").getItem(0))
        .withColumn("prob_positive", F.col("probability").getItem(1))
        .withColumn(
            "confidence",
            F.greatest(
                F.col("probability").getItem(0),
                F.col("probability").getItem(1)
            )
        )
        .withColumn("processing_time", F.current_timestamp())
        .select(
            "text",
            F.coalesce(F.col("product"), F.lit("Apple General")).alias("product"),
            "annotation",
            "prob_negative",
            "prob_positive",
            "confidence",
            "sentiment_label",
            "processing_time",
            "kafka_timestamp"
        )
    )
    return enriched


def write_apple_predictions(enriched_stream, output_path: Path, checkpoint_path: Path):
    """Écriture des prédictions Apple en Parquet (append)."""
    return (
        enriched_stream.writeStream
        .format("parquet")
        .outputMode("append")
        .option("path", str(output_path))
        .option("checkpointLocation", str(checkpoint_path))
        .trigger(processingTime=f"{STREAMING_TRIGGER_SECONDS} seconds")
        .start()
    )


def compute_bad_buzz_alerts(enriched_stream):
    """
    Agrégation en fenêtre glissante pour la détection de bad buzz.
    Requête SQL sur le flux Apple.
    """
    window_duration = f"{BAD_BUZZ_WINDOW_MINUTES} minutes"
    slide_duration = "1 minute"

    agg = (
        enriched_stream
        .withWatermark("processing_time", window_duration)
        .groupBy(
            F.window(F.col("processing_time"), window_duration, slide_duration),
            F.col("product")
        )
        .agg(
            F.count("*").alias("total_tweets"),
            F.sum(F.when(F.col("sentiment_label") == "Positive", 1).otherwise(0))
             .alias("positive_count"),
            F.sum(F.when(F.col("sentiment_label") == "Negative", 1).otherwise(0))
             .alias("negative_count"),
            F.sum(F.when(F.col("sentiment_label") == "Neutral", 1).otherwise(0))
             .alias("neutral_count"),
            F.avg("confidence").alias("avg_confidence")
        )
        .withColumn(
            "neg_ratio",
            F.col("negative_count") /
            F.nullif(F.col("positive_count") + F.col("negative_count"), 0)
        )
        .withColumn(
            "reputation_score",
            F.col("positive_count") * 100.0 /
            F.nullif(F.col("positive_count") + F.col("negative_count"), 0)
        )
        .withColumn(
            "status",
            F.when(F.col("neg_ratio") > BAD_BUZZ_THRESHOLD, "BAD_BUZZ")
             .when(F.col("neg_ratio") > 0.50, "SURVEILLANCE")
             .otherwise("STABLE")
        )
    )
    return agg


def run_apple_consumer():
    """Pipeline complet du consommateur Apple Phase 3."""
    ensure_dirs(APPLE_PREDICTIONS_DIR, CHECKPOINT_DIR_APPLE)

    spark = get_spark_kafka_session("SentimentStreaming_Phase3_AppleMonitoring")

    try:
        # ── Chargement du modèle Phase 1 ──────────────────────────────────────
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle non trouvé : {MODEL_PATH}\n"
                "Lancez d'abord : python src/train_model.py"
            )
        pipeline_model = PipelineModel.load(str(MODEL_PATH))
        logger.info("Modèle ML Phase 1 chargé pour Phase 3")

        # ── Flux Kafka Apple ──────────────────────────────────────────────────
        raw_stream = create_apple_kafka_stream(
            spark, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_APPLE
        )

        # ── Parsing + ML + Neutral ────────────────────────────────────────────
        parsed_stream = parse_apple_messages(raw_stream)
        enriched_stream = apply_ml_and_neutral(parsed_stream, pipeline_model)

        # ── Écriture prédictions Parquet ──────────────────────────────────────
        logger.info(f"Prédictions Apple → {APPLE_PREDICTIONS_DIR}")
        query_predictions = write_apple_predictions(
            enriched_stream, APPLE_PREDICTIONS_DIR, CHECKPOINT_DIR_APPLE
        )

        # ── Bad Buzz en console (monitoring temps réel) ───────────────────────
        bad_buzz_stream = compute_bad_buzz_alerts(enriched_stream)
        query_buzz = (
            bad_buzz_stream.writeStream
            .outputMode("update")
            .format("console")
            .option("truncate", "false")
            .option("numRows", 20)
            .trigger(processingTime=f"{STREAMING_TRIGGER_SECONDS} seconds")
            .start()
        )

        logger.info(
            "Phase 3 — Apple Brand Monitoring démarré\n"
            f"  Topic       : {KAFKA_TOPIC_APPLE}\n"
            f"  Seuil conf  : {CONFIDENCE_THRESHOLD}\n"
            f"  Seuil buzz  : {BAD_BUZZ_THRESHOLD * 100:.0f}%\n"
            f"  Fenêtre     : {BAD_BUZZ_WINDOW_MINUTES} minutes\n"
            "  CTRL+C pour arrêter"
        )

        import time
        while query_predictions.isActive:
            time.sleep(STREAMING_TRIGGER_SECONDS)

    except KeyboardInterrupt:
        logger.warning("Consumer Apple interrompu (CTRL+C)")
    finally:
        if 'query_predictions' in locals():
            query_predictions.stop()
        if 'query_buzz' in locals():
            query_buzz.stop()
        spark.stop()
        logger.info("Phase 3 terminée — données Apple sauvegardées")


if __name__ == "__main__":
    run_apple_consumer()