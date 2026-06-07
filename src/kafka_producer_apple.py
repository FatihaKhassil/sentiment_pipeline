"""
src/kafka_producer_apple.py
Phase 3 — Producteur Kafka pour Apple Tweets (Brand Monitoring).
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
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_APPLE,
    KAFKA_PRODUCER_DELAY_APPLE, APPLE_CSV,
    APPLE_TEXT_COL, APPLE_PRODUCT_COL, APPLE_LABEL_COL
)
from src.utils import get_logger

logger = get_logger(__name__)


def run_apple_producer(
    csv_path: str = str(APPLE_CSV),
    topic: str = KAFKA_TOPIC_APPLE,
    delay: float = KAFKA_PRODUCER_DELAY_APPLE,
    loop: bool = False,
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS
):
    if not Path(csv_path).exists():
        logger.error(f"Fichier non trouve : {csv_path}")
        logger.error("Placez apple_tweets.csv dans data/raw/")
        sys.exit(1)

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        acks=1,
        retries=3,
        linger_ms=10
    )
    logger.info(f"Producteur Apple connecte -> {bootstrap_servers}")
    logger.info(f"  Topic : {topic} | Delai : {delay*1000:.0f}ms | Boucle : {loop}")

    run = True
    iteration = 0

    try:
        while run:
            iteration += 1
            sent = 0

            with open(csv_path, encoding="utf-8", newline="", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tweet_text = row.get(APPLE_TEXT_COL, "").strip()
                    if not tweet_text or tweet_text.lower() in ("nan", "null", ""):
                        continue

                    # Simplification du nom de produit
                    product = str(row.get(APPLE_PRODUCT_COL, "")).strip()
                    for key in ["iPhone", "iPad", "iOS", "MacBook", "Apple"]:
                        if key.lower() in product.lower():
                            product = key
                            break
                    else:
                        product = "Apple General"

                    message = {
                        "tweet_text": tweet_text,
                        "product": product,
                        "annotation": str(row.get(APPLE_LABEL_COL, "")),
                        "ingestion_timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                    producer.send(topic, value=message)
                    sent += 1

                    if sent % 500 == 0:
                        logger.info(f"  Apple tweets envoyes : {sent:,}")

                    if delay > 0:
                        time.sleep(delay)

            logger.info(f"Iteration {iteration} terminee — {sent:,} tweets envoyes")

            if not loop:
                run = False
            else:
                time.sleep(2)

    except KeyboardInterrupt:
        logger.warning("Producteur Apple interrompu (CTRL+C)")
    finally:
        producer.flush()
        producer.close()
        logger.info("Producteur Apple ferme")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(APPLE_CSV))
    parser.add_argument("--topic", default=KAFKA_TOPIC_APPLE)
    parser.add_argument("--delay", type=float, default=KAFKA_PRODUCER_DELAY_APPLE)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS)
    args = parser.parse_args()

    run_apple_producer(
        csv_path=args.csv,
        topic=args.topic,
        delay=args.delay,
        loop=args.loop,
        bootstrap_servers=args.bootstrap_servers
    )