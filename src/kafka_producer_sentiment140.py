"""
src/kafka_producer_sentiment140.py
Phase 2 — Producteur Kafka pour Sentiment140 (simulation flux Big Data).
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
from config.config import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_SENTIMENT140,
    KAFKA_PRODUCER_DELAY_S140, SENTIMENT140_CSV, S140_COLUMNS
)
from src.utils import get_logger

logger = get_logger(__name__)


def run_producer(
    csv_path: str = str(SENTIMENT140_CSV),
    topic: str = KAFKA_TOPIC_SENTIMENT140,
    delay: float = KAFKA_PRODUCER_DELAY_S140,
    max_rows: int = None,
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS
):
    if not Path(csv_path).exists():
        logger.error(f"Fichier non trouve : {csv_path}")
        sys.exit(1)

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        acks="all",
        retries=3,
        batch_size=16384,
        linger_ms=5,
        compression_type="gzip",
        max_request_size=1048576
    )
    logger.info(f"Producteur Kafka connecte -> {bootstrap_servers}")
    logger.info(f"  Topic  : {topic}")
    logger.info(f"  Delai  : {delay * 1000:.1f} ms/tweet -> ~{1/delay:.0f} tweets/s")
    logger.info(f"  Limite : {'Aucune (1.6M tweets)' if max_rows is None else f'{max_rows:,} tweets'}")

    sent = 0
    start_time = time.time()
    last_log_time = start_time

    try:
        with open(csv_path, encoding="latin-1", newline="") as f:
            reader = csv.DictReader(f, fieldnames=S140_COLUMNS)
            for row in reader:
                if max_rows and sent >= max_rows:
                    break
                if not row.get("text", "").strip():
                    continue

                message = {
                    "text": row.get("text", ""),
                    "target": int(row.get("target", 0)),
                    "id": row.get("id", ""),
                    "ingestion_timestamp": datetime.utcnow().isoformat() + "Z"
                }
                producer.send(topic, value=message)
                sent += 1

                now = time.time()
                if sent % 100_000 == 0 or (now - last_log_time) > 30:
                    elapsed = now - start_time
                    rate = sent / elapsed
                    eta = (1_600_000 - sent) / rate if rate > 0 else 0
                    logger.info(f"  Envoyes : {sent:>8,} | Debit : {rate:>6.0f} tweets/s | ETA : {eta/60:.1f} min")
                    last_log_time = now

                if delay > 0:
                    time.sleep(delay)

    except KeyboardInterrupt:
        logger.warning("Producteur interrompu (CTRL+C)")
    finally:
        producer.flush()
        producer.close()
        elapsed = time.time() - start_time
        logger.info(f"Tweets envoyes : {sent:,} en {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(SENTIMENT140_CSV))
    parser.add_argument("--topic", default=KAFKA_TOPIC_SENTIMENT140)
    parser.add_argument("--delay", type=float, default=KAFKA_PRODUCER_DELAY_S140)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS)
    args = parser.parse_args()

    run_producer(
        csv_path=args.csv,
        topic=args.topic,
        delay=args.delay,
        max_rows=args.max_rows,
        bootstrap_servers=args.bootstrap_servers
    )