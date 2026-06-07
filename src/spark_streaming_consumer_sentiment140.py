"""
src/spark_streaming_consumer_sentiment140.py
Phase 2 — Consommateur Spark Structured Streaming pour Sentiment140.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import PipelineModel
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

from config.config import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_SENTIMENT140,
    MODEL_PATH, SENTIMENT140_PREDICTIONS_DIR, CHECKPOINT_DIR_S140,
    STREAMING_TRIGGER_SECONDS, STREAMING_MAX_OFFSETS,
    CONFIDENCE_THRESHOLD, SPARK_MASTER,
    SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY
)
from src.utils import get_logger, ensure_dirs, clean_text_udf, get_neutral_udf

logger = get_logger(__name__)

KAFKA_MESSAGE_SCHEMA = StructType([
    StructField("text", StringType(), True),
    StructField("target", IntegerType(), True),
    StructField("id", StringType(), True),
    StructField("ingestion_timestamp", StringType(), True)
])


def run_streaming_consumer():
    ensure_dirs(SENTIMENT140_PREDICTIONS_DIR, CHECKPOINT_DIR_S140)

    # Import PySpark ici pour eviter les problemes d'initialisation
    from pyspark.sql import SparkSession

    kafka_package = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"

    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName("SentimentStreaming_Phase2")
        .config("spark.jars.packages", kafka_package)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"SparkSession creee — Master: {spark.sparkContext.master}")

    # Charger le modele
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele non trouve : {MODEL_PATH}\nLancez d'abord le notebook 03")
    pipeline_model = PipelineModel.load(str(MODEL_PATH))
    logger.info("Modele ML charge")

    # UDFs
    prob_neg_udf = udf(lambda v: float(v[0]), DoubleType())
    prob_pos_udf = udf(lambda v: float(v[1]), DoubleType())
    neutral_udf  = get_neutral_udf(CONFIDENCE_THRESHOLD)

    # Flux Kafka
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC_SENTIMENT140)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", STREAMING_MAX_OFFSETS)
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parsing JSON
    parsed = (
        raw_stream
        .select(
            F.from_json(F.col("value").cast("string"), KAFKA_MESSAGE_SCHEMA).alias("data"),
            F.col("timestamp").alias("kafka_timestamp")
        )
        .select("data.*", "kafka_timestamp")
        .filter(F.col("text").isNotNull() & (F.length(F.col("text")) > 0))
    )

    # Nettoyage + inference ML
    cleaned = parsed.withColumn("text", clean_text_udf(F.col("text")))
    cleaned = cleaned.filter(F.length(F.col("text")) > 2)
    predictions = pipeline_model.transform(cleaned)

    # Extraction probabilites + couche Neutral
    enriched = (
        predictions
        .withColumn("prob_negative", prob_neg_udf(F.col("probability")))
        .withColumn("prob_positive", prob_pos_udf(F.col("probability")))
        .withColumn("confidence", F.greatest(F.col("prob_negative"), F.col("prob_positive")))
        .withColumn(
            "sentiment_label",
            neutral_udf(F.col("prediction"), F.col("prob_negative"), F.col("prob_positive"))
        )
        .withColumn("processing_time", F.current_timestamp())
        .select(
            "text",
            F.col("target").alias("original_target"),
            F.col("prediction").alias("ml_prediction"),
            "prob_negative",
            "prob_positive",
            "confidence",
            "sentiment_label",
            "processing_time",
            "kafka_timestamp"
        )
    )

    # Ecriture Parquet
    logger.info(f"Ecriture Parquet -> {SENTIMENT140_PREDICTIONS_DIR}")
    query = (
        enriched.writeStream
        .format("parquet")
        .outputMode("append")
        .option("path", str(SENTIMENT140_PREDICTIONS_DIR))
        .option("checkpointLocation", str(CHECKPOINT_DIR_S140))
        .trigger(processingTime=f"{STREAMING_TRIGGER_SECONDS} seconds")
        .start()
    )

    logger.info(f"Streaming demarre — micro-batch toutes les {STREAMING_TRIGGER_SECONDS}s")
    logger.info("CTRL+C pour arreter proprement")

    try:
        import time
        while query.isActive:
            progress = query.lastProgress
            if progress:
                n = progress.get("numInputRows", 0)
                if n > 0:
                    logger.info(f"Micro-batch traite : {n:,} tweets")
            time.sleep(STREAMING_TRIGGER_SECONDS)
    except KeyboardInterrupt:
        logger.warning("Consumer interrompu (CTRL+C)")
        query.stop()
    finally:
        spark.stop()
        logger.info("SparkSession fermee")


if __name__ == "__main__":
    run_streaming_consumer()