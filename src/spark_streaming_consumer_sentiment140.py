"""
src/kafka_producer_sentiment140.py
Phase 2 — Producteur Kafka pour Sentiment140 (simulation flux Big Data).

Lit le fichier sentiment140.csv ligne par ligne et publie chaque tweet
dans le topic Kafka 'sentiment_stream' à un débit configurable.

Débit par défaut : 1 000 tweets/s → 1,6M tweets en ~27 minutes

Usage :
    python src/kafka_producer_sentiment140.py
    python src/kafka_producer_sentiment140.py --delay 0.005 --max-rows 100000
"""

import sys
import json
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from kafka import KafkaProducer
from kafka.errors import KafkaError

from config.config import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_SENTIMENT140,
    KAFKA_PRODUCER_DELAY_S140, SENTIMENT140_CSV, S140_COLUMNS
)
from src.utils import get_logger

logger = get_logger(__name__)


def create_producer(bootstrap_servers: str) -> KafkaProducer:
    """Crée et retourne un KafkaProducer configuré."""
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        acks="all",                  # Attente confirmation du broker
        retries=3,
        batch_size=16384,            # 16 KB par batch
        linger_ms=5,                 # Attente 5ms pour regrouper les messages
        compression_type="gzip",     # Compression pour réduire le trafic réseau
        max_request_size=1048576     # 1 MB max par requête
    )
    logger.info(f"Producteur Kafka connecté → {bootstrap_servers}")
    return producer


def build_message(row: dict) -> dict:
    """
    Construit le message JSON à partir d'une ligne CSV Sentiment140.

    Structure du message :
        {
            "text": str,                    # Feature ML principale
            "target": int,                  # Label original (0=négatif, 4=positif)
            "id": str,                      # ID du tweet
            "ingestion_timestamp": str      # Timestamp d'ingestion (ISO 8601)
        }
    """
    return {
        "text": row.get("text", ""),
        "target": int(row.get("target", 0)),
        "id": row.get("id", ""),
        "ingestion_timestamp": datetime.utcnow().isoformat() + "Z"
    }


def on_send_success(record_metadata):
    """Callback appelé en cas de succès d'envoi."""
    pass  # Silencieux pour les performances — activer pour debug


def on_send_error(exc):
    """Callback appelé en cas d'erreur d'envoi."""
    logger.error(f"Erreur envoi Kafka : {exc}")


def run_producer(
    csv_path: str = str(SENTIMENT140_CSV),
    topic: str = KAFKA_TOPIC_SENTIMENT140,
    delay: float = KAFKA_PRODUCER_DELAY_S140,
    max_rows: int = None,
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS
):
    """
    Pipeline principal du producteur Kafka Sentiment140.

    Args:
        csv_path: Chemin vers sentiment140.csv
        topic: Nom du topic Kafka cible
        delay: Délai entre messages en secondes (défaut: 0.001 = 1ms)
        max_rows: Limite du nombre de tweets (None = tous les 1,6M)
        bootstrap_servers: Adresse du broker Kafka
    """
    if not Path(csv_path).exists():
        logger.error(f"Fichier non trouvé : {csv_path}")
        logger.error("Téléchargez Sentiment140 et placez-le dans data/raw/")
        sys.exit(1)

    producer = create_producer(bootstrap_servers)

    sent = 0
    errors = 0
    start_time = time.time()
    last_log_time = start_time

    logger.info(f"Démarrage du producteur Kafka")
    logger.info(f"  Topic   : {topic}")
    logger.info(f"  Délai   : {delay * 1000:.1f} ms/tweet → ~{1/delay:.0f} tweets/s")
    logger.info(f"  Limite  : {'Aucune (1,6M tweets)' if max_rows is None else f'{max_rows:,} tweets'}")

    try:
        with open(csv_path, encoding="latin-1", newline="") as f:
            reader = csv.DictReader(f, fieldnames=S140_COLUMNS)

            for row in reader:
                if max_rows and sent >= max_rows:
                    break

                # Ignorer les lignes vides
                if not row.get("text", "").strip():
                    continue

                message = build_message(row)

                # Envoi asynchrone avec callbacks
                producer.send(topic, value=message) \
                    .add_callback(on_send_success) \
                    .add_errback(on_send_error)

                sent += 1

                # Log de progression toutes les 100 000 lignes ou 30s
                now = time.time()
                if sent % 100_000 == 0 or (now - last_log_time) > 30:
                    elapsed = now - start_time
                    rate = sent / elapsed
                    eta = (1_600_000 - sent) / rate if rate > 0 else 0
                    logger.info(
                        f"  Envoyés : {sent:>8,} | "
                        f"Débit : {rate:>6.0f} tweets/s | "
                        f"ETA : {eta/60:.1f} min"
                    )
                    last_log_time = now

                if delay > 0:
                    time.sleep(delay)

    except KeyboardInterrupt:
        logger.warning("\nProducteur interrompu par l'utilisateur (CTRL+C)")
    finally:
        producer.flush()  # Garantit l'envoi de tous les messages en attente
        producer.close()

        elapsed = time.time() - start_time
        logger.info(f"\n── Résumé du producteur ──")
        logger.info(f"  Tweets envoyés  : {sent:,}")
        logger.info(f"  Erreurs         : {errors:,}")
        logger.info(f"  Durée totale    : {elapsed/60:.1f} minutes")
        logger.info(f"  Débit moyen     : {sent/elapsed:.0f} tweets/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Producteur Kafka — Sentiment140 Big Data Stream"
    )
    parser.add_argument("--csv", default=str(SENTIMENT140_CSV),
                        help="Chemin vers sentiment140.csv")
    parser.add_argument("--topic", default=KAFKA_TOPIC_SENTIMENT140,
                        help="Topic Kafka cible")
    parser.add_argument("--delay", type=float, default=KAFKA_PRODUCER_DELAY_S140,
                        help="Délai entre messages (secondes)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Nombre maximum de tweets à envoyer")
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS,
                        help="Adresse du broker Kafka")
    args = parser.parse_args()

    run_producer(
        csv_path=args.csv,
        topic=args.topic,
        delay=args.delay,
        max_rows=args.max_rows,
        bootstrap_servers=args.bootstrap_servers
    )