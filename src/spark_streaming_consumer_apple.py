"""
src/spark_streaming_consumer_apple.py
Phase 3 — Consommateur Spark Structured Streaming pour Apple Brand Monitoring.
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import udf
from pyspark.ml import PipelineModel

from config.config import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_APPLE,
    MODEL_PATH, APPLE_PREDICTIONS_DIR, CHECKPOINT_DIR_APPLE,
    STREAMING_TRIGGER_SECONDS, STREAMING_MAX_OFFSETS,
    CONFIDENCE_THRESHOLD, BAD_BUZZ_THRESHOLD, BAD_BUZZ_WINDOW_MINUTES,
    SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY
)
from src.utils import ensure_dirs, get_logger, clean_text_udf, get_neutral_udf

logger = get_logger(__name__)

APPLE_MESSAGE_SCHEMA = StructType([
    StructField("tweet_text", StringType(), True),
    StructField("product", StringType(), True),
    StructField("annotation", StringType(), True),
    StructField("ingestion_timestamp", StringType(), True)
])


def run_apple_consumer():
    ensure_dirs(APPLE_PREDICTIONS_DIR, CHECKPOINT_DIR_APPLE)

    from pyspark.sql import SparkSession

    kafka_package = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"

    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName("SentimentStreaming_Phase3_Apple")
        .config("spark.jars.packages", kafka_package)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Charger le modele
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele non trouve : {MODEL_PATH}")
    pipeline_model = PipelineModel.load(str(MODEL_PATH))
    logger.info("Modele ML Phase 1 charge pour Phase 3")

    # UDFs extraction probabilites (evite getItem qui echoue sur vecteur sparse)
    prob_neg_udf = udf(lambda v: float(v[0]), DoubleType())
    prob_pos_udf = udf(lambda v: float(v[1]), DoubleType())
    neutral_udf  = get_neutral_udf(CONFIDENCE_THRESHOLD)

    # Flux Kafka
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC_APPLE)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", STREAMING_MAX_OFFSETS)
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parsing + renommage tweet_text -> text
    parsed = (
        raw_stream
        .select(
            F.from_json(F.col("value").cast("string"), APPLE_MESSAGE_SCHEMA).alias("data"),
            F.col("timestamp").alias("kafka_timestamp")
        )
        .select("data.*", "kafka_timestamp")
        .withColumnRenamed("tweet_text", "text")
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

    # Ecriture Parquet
    logger.info(f"Predictions Apple -> {APPLE_PREDICTIONS_DIR}")
    query = (
        enriched.writeStream
        .format("parquet")
        .outputMode("append")
        .option("path", str(APPLE_PREDICTIONS_DIR))
        .option("checkpointLocation", str(CHECKPOINT_DIR_APPLE))
        .trigger(processingTime=f"{STREAMING_TRIGGER_SECONDS} seconds")
        .start()
    )

    logger.info(f"Phase 3 demarre — Topic: {KAFKA_TOPIC_APPLE} | CTRL+C pour arreter")

    try:
        while query.isActive:
            progress = query.lastProgress
            if progress:
                n = progress.get("numInputRows", 0)
                if n > 0:
                    logger.info(f"Apple tweets traites : {n:,}")
            time.sleep(STREAMING_TRIGGER_SECONDS)
    except KeyboardInterrupt:
        logger.warning("Consumer Apple interrompu (CTRL+C)")
        query.stop()
    finally:
        spark.stop()
        logger.info("Phase 3 terminee")


if __name__ == "__main__":
    run_apple_consumer()