"""
config/config.py
Centralise tous les paramètres du projet Big Data Sentiment Analysis.
Importer ce fichier dans tous les scripts et notebooks.
"""

import os
from pathlib import Path

# ─── Chemins racine ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

# ─── Données brutes ──────────────────────────────────────────────────────────
RAW_DIR             = DATA_DIR / "raw"
SENTIMENT140_CSV    = RAW_DIR / "sentiment140.csv"
APPLE_CSV           = RAW_DIR / "apple_tweets.csv"

# ─── Données traitées ────────────────────────────────────────────────────────
PROCESSED_DIR               = DATA_DIR / "processed"
SENTIMENT140_CLEAN_PARQUET  = PROCESSED_DIR / "sentiment140_clean.parquet"

# ─── Modèles ─────────────────────────────────────────────────────────────────
MODELS_DIR      = DATA_DIR / "models"
MODEL_PATH      = MODELS_DIR / "sentiment_model"

# ─── Streaming ───────────────────────────────────────────────────────────────
STREAMING_DIR                  = DATA_DIR / "streaming"
SENTIMENT140_PREDICTIONS_DIR   = STREAMING_DIR / "sentiment140_predictions"
APPLE_PREDICTIONS_DIR          = STREAMING_DIR / "apple_predictions"

# ─── Métriques ───────────────────────────────────────────────────────────────
METRICS_DIR             = DATA_DIR / "metrics"
SENTIMENT_METRICS_PATH  = METRICS_DIR / "sentiment_metrics.parquet"
APPLE_METRICS_PATH      = METRICS_DIR / "apple_brand_stats.parquet"

# ─── Checkpoints Spark Streaming ─────────────────────────────────────────────
CHECKPOINT_DIR_S140     = ROOT_DIR / "checkpoint" / "sentiment140"
CHECKPOINT_DIR_APPLE    = ROOT_DIR / "checkpoint" / "apple"

# ─── Kafka ───────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_SENTIMENT140    = "sentiment_stream"
KAFKA_TOPIC_APPLE           = "apple_tweets"
KAFKA_PRODUCER_DELAY_S140   = 0.001   # 1 ms → ~1 000 tweets/s
KAFKA_PRODUCER_DELAY_APPLE  = 0.05    # 50 ms → ~20 tweets/s

# ─── Spark ───────────────────────────────────────────────────────────────────
SPARK_DRIVER_MEMORY     = "4g"
SPARK_EXECUTOR_MEMORY   = "4g"
SPARK_SHUFFLE_PARTITIONS = "8"
SPARK_MASTER            = "local[*]"

# ─── Spark Streaming ─────────────────────────────────────────────────────────
STREAMING_TRIGGER_SECONDS   = 10   # micro-batch toutes les 10 s
STREAMING_MAX_OFFSETS       = 10000  # max messages par micro-batch

# ─── NLP Pipeline ────────────────────────────────────────────────────────────
TFIDF_NUM_FEATURES  = 2 ** 18   # 262 144 features TF-IDF
STOPWORDS_LANG      = "english"

# ─── Modèle ML ───────────────────────────────────────────────────────────────
TRAIN_RATIO             = 0.8
TEST_RATIO              = 0.2
RANDOM_SEED             = 42
LR_MAX_ITER             = 100
LR_REG_PARAM            = 0.01

# ─── Logique métier Neutral ───────────────────────────────────────────────────
CONFIDENCE_THRESHOLD    = 0.60   # seuil de confiance pour Pos/Neg
BAD_BUZZ_THRESHOLD      = 0.70   # seuil de négativité pour alerte
BAD_BUZZ_WINDOW_MINUTES = 5      # fenêtre glissante

# ─── Dashboard ───────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_SECONDS   = 10
DASHBOARD_TITLE             = "Real-Time Sentiment Analysis — Brand Monitoring"
TOP_NEGATIVE_TWEETS_N       = 10

# ─── Sentiment140 colonnes ───────────────────────────────────────────────────
S140_COLUMNS        = ["target", "id", "date", "flag", "user", "text"]
S140_TARGET_COL     = "target"
S140_TEXT_COL       = "text"

# ─── Apple Tweets colonnes ────────────────────────────────────────────────────
APPLE_TEXT_COL      = "tweet_text"
APPLE_PRODUCT_COL   = "emotion_in_tweet_is_directed_at"
APPLE_LABEL_COL     = "is_there_an_emotion_directed_at_a_brand_or_product"