"""
src/utils.py
Fonctions utilitaires partagées par tous les modules du projet.
"""

import re
import sys
import logging
from pathlib import Path
from datetime import datetime

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# Ajouter le répertoire racine au path Python
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config.config import (
    SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY,
    SPARK_SHUFFLE_PARTITIONS, CONFIDENCE_THRESHOLD
)

# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Retourne un logger configuré avec format uniforme."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

logger = get_logger(__name__)

# ─── Spark Session ────────────────────────────────────────────────────────────

def get_spark_session(app_name: str, extra_configs: dict = None) -> SparkSession:
    """
    Crée ou récupère une SparkSession configurée pour le projet.

    Args:
        app_name: Nom de l'application Spark
        extra_configs: Configurations Spark supplémentaires

    Returns:
        SparkSession active
    """
    builder = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.adaptive.enabled", "true")
        # Parquet optimizations
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.parquet.writeLegacyFormat", "false")
    )

    if extra_configs:
        for key, value in extra_configs.items():
            builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"SparkSession '{app_name}' démarrée — version Spark {spark.version}")
    return spark


def get_spark_kafka_session(app_name: str) -> SparkSession:
    """
    SparkSession avec connecteur Kafka intégré (jar téléchargé automatiquement).
    Utilise les packages Maven pour kafka-clients.
    """
    kafka_package = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
    return get_spark_session(
        app_name,
        extra_configs={"spark.jars.packages": kafka_package}
    )

# ─── NLP Nettoyage texte ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Nettoyage textuel : URLs, @mentions, hashtags, ponctuation, casse.
    Utilisé en UDF Spark et en preprocessing Python natif.
    """
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # Suppression URLs
    text = re.sub(r"@\w+", "", text)                  # Suppression @mentions
    text = re.sub(r"#", "", text)                     # Suppression symbole #
    text = re.sub(r"[^a-z\s]", "", text)              # Garde uniquement lettres
    text = re.sub(r"\s+", " ", text).strip()          # Normalise espaces
    return text


# UDF Spark pour le nettoyage de texte
clean_text_udf = F.udf(clean_text, StringType())

# ─── Logique Métier Neutral ───────────────────────────────────────────────────

def apply_neutral_rule(prediction, prob_neg, prob_pos,
                       threshold: float = CONFIDENCE_THRESHOLD) -> str:
    """
    Convertit la prédiction binaire ML en étiquette métier à 3 états.

    Args:
        prediction: 0.0 (négatif) ou 1.0 (positif) — sortie Logistic Regression
        prob_neg: P(négatif) — probability[0]
        prob_pos: P(positif) — probability[1]
        threshold: seuil de confiance (défaut : 0.60)

    Returns:
        "Positive", "Negative", ou "Neutral"
    """
    if prediction is None or prob_neg is None or prob_pos is None:
        return "Neutral"
    confidence = max(prob_neg, prob_pos)
    if confidence < threshold:
        return "Neutral"
    return "Positive" if prediction == 1.0 else "Negative"


def get_neutral_udf(threshold: float = CONFIDENCE_THRESHOLD):
    """Retourne un UDF Spark configuré avec le seuil de confiance."""
    from pyspark.sql.functions import udf
    def _apply(prediction, prob_neg, prob_pos):
        return apply_neutral_rule(prediction, prob_neg, prob_pos, threshold)
    return udf(_apply, StringType())

# ─── Score de réputation ──────────────────────────────────────────────────────

def compute_reputation_score(n_positive: int, n_negative: int) -> float:
    """
    Score de réputation [0, 100] : Positifs / (Positifs + Négatifs) × 100.
    Les tweets Neutrals sont exclus du calcul.
    """
    total = n_positive + n_negative
    if total == 0:
        return 50.0   # Score neutre par défaut
    return round((n_positive / total) * 100, 2)


def get_bad_buzz_status(n_negative: int, n_positive: int,
                        threshold: float = 0.70) -> str:
    """
    Statut de la marque selon le taux de négativité.

    Returns:
        "BAD_BUZZ", "SURVEILLANCE", ou "STABLE"
    """
    total = n_positive + n_negative
    if total == 0:
        return "STABLE"
    neg_ratio = n_negative / total
    if neg_ratio > threshold:
        return "BAD_BUZZ"
    elif neg_ratio > 0.50:
        return "SURVEILLANCE"
    return "STABLE"

# ─── Utilitaires fichiers ─────────────────────────────────────────────────────

def ensure_dirs(*paths):
    """Crée les répertoires s'ils n'existent pas."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Répertoire assuré : {p}")


def parquet_exists(path) -> bool:
    """Vérifie si un répertoire Parquet non-vide existe."""
    p = Path(path)
    if not p.exists():
        return False
    parquet_files = list(p.glob("*.parquet"))
    return len(parquet_files) > 0


def format_number(n: int) -> str:
    """Formate un entier avec séparateurs de milliers."""
    return f"{n:,}"


def now_str() -> str:
    """Retourne l'horodatage courant formaté."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")