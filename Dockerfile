FROM apache/spark:3.5.1

USER root

# ── Dépendances système ──────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# ── PySpark : déjà dans l'image, juste ajouter au PYTHONPATH ────
# NE PAS pip install pyspark — évite le téléchargement de 317 MB

# ── Autres dépendances (petites, pas de timeout) ─────────────────
COPY requirements-spark.txt /tmp/requirements-spark.txt
RUN pip3 install --no-cache-dir --timeout 300 --retries 10 -r /tmp/requirements-spark.txt

RUN mkdir -p /workspace
WORKDIR /workspace

# ── PySpark depuis l'image de base (pas de téléchargement) ───────
ENV SPARK_HOME=/opt/spark
ENV PYTHONPATH=/workspace:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PATH=$PATH:/opt/spark/bin:/usr/local/bin

USER root