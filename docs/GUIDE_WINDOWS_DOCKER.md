# Guide d'exécution Windows — Stack Docker Spark + Kafka

## Ce guide remplace complètement l'installation locale Spark/Java/Hadoop sur Windows

---

## PRÉREQUIS (à installer une seule fois)

### 1. Docker Desktop pour Windows
- Télécharger : https://www.docker.com/products/docker-desktop/
- Installer et redémarrer
- Activer WSL2 backend (recommandé) dans Settings → General
- Vérifier : ouvrir PowerShell et taper `docker --version`

### 2. Python 3.10 sur Windows (pour Streamlit + Kafka producers)
- Télécharger : https://www.python.org/downloads/release/python-31011/
- Cocher "Add Python to PATH" lors de l'installation
- Vérifier : `python --version`

### 3. Environnement virtuel Python (dans le dossier du projet)
```powershell
# Ouvrir PowerShell dans le dossier du projet
cd C:\chemin\vers\sentiment-pipeline

python -m venv venv
.\venv\Scripts\activate

# Installer uniquement les dépendances Windows (pas PySpark local !)
pip install kafka-python streamlit plotly pandas pyarrow python-dotenv tqdm
```

> NE PAS installer pyspark dans le venv Windows.
> PySpark tourne entièrement dans Docker.

---

## STRUCTURE APRÈS MIGRATION

```
sentiment-pipeline/
├── docker-compose.yml      ← MODIFIÉ (Spark + Kafka)
├── Dockerfile              ← NOUVEAU (image Spark custom)
├── requirements-spark.txt  ← NOUVEAU (dépendances dans Docker)
├── config/
│   └── config.py           ← MODIFIÉ (détection Docker/Windows)
├── src/                    ← INCHANGÉ
├── notebooks/              ← MODIFIER findspark (voir section ci-dessous)
├── dashboard/              ← INCHANGÉ
└── data/                   ← INCHANGÉ (monté dans Docker)
```

---

## ÉTAPE 1 — Modifications dans les notebooks

### Supprimer findspark de TOUS les notebooks

Chercher et supprimer ces lignes dans chaque notebook :
```python
# SUPPRIMER ces lignes
import findspark
findspark.init()
```

### Adapter la SparkSession dans chaque notebook

**AVANT (version Windows locale) :**
```python
import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("MonApp") \
    .getOrCreate()
```

**APRÈS (version Docker) :**
```python
# Plus de findspark !
import sys
from pathlib import Path
sys.path.insert(0, str(Path().absolute().parent))

from pyspark.sql import SparkSession
from config.config import SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY

spark = SparkSession.builder \
    .master(SPARK_MASTER) \
    .appName("MonApp") \
    .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
    .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark version :", spark.version)
print("Master :", spark.sparkContext.master)
```

### Pour les notebooks avec Kafka (streaming)

**AVANT :**
```python
spark = SparkSession.builder \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .getOrCreate()
```

**APRÈS (le jar Kafka est déjà dans l'image Docker) :**
```python
from config.config import SPARK_MASTER, KAFKA_BOOTSTRAP_SERVERS

spark = SparkSession.builder \
    .master(SPARK_MASTER) \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .getOrCreate()

# KAFKA_BOOTSTRAP_SERVERS vaut automatiquement :
# - "kafka:29092"    si le script tourne dans Docker
# - "localhost:9092" si le script tourne sur Windows
print("Kafka servers :", KAFKA_BOOTSTRAP_SERVERS)
```

---

## ÉTAPE 2 — Construction et démarrage de la stack

### Ouvrir PowerShell dans le dossier du projet

```powershell
cd C:\chemin\vers\sentiment-pipeline
```

### Construire l'image Docker Spark (une seule fois, ~3-5 minutes)

```powershell
docker-compose build
```

Vous devriez voir : `Successfully built xxxx` et `Successfully tagged sentiment-spark:latest`

### Démarrer la stack complète

```powershell
docker-compose up -d
```

### Vérifier que tout est UP

```powershell
docker-compose ps
```

Résultat attendu :
```
NAME            STATUS          PORTS
zookeeper       running (healthy)   0.0.0.0:2181->2181/tcp
kafka           running (healthy)   0.0.0.0:9092->9092/tcp
kafka-ui        running             0.0.0.0:8080->8080/tcp
spark-master    running (healthy)   0.0.0.0:7077->7077/tcp, 0.0.0.0:9090->9090/tcp
spark-worker    running             
```

### Interfaces web disponibles

| Interface | URL | Rôle |
|-----------|-----|------|
| Kafka UI | http://localhost:8080 | Voir les topics et messages |
| Spark Master | http://localhost:9090 | Voir les jobs Spark |

---

## ÉTAPE 3 — Créer les topics Kafka

```powershell
# Topic pour Sentiment140 (Phase 2)
docker exec kafka kafka-topics --create `
    --bootstrap-server localhost:9092 `
    --topic sentiment_stream `
    --partitions 4 `
    --replication-factor 1 `
    --if-not-exists

# Topic pour Apple Tweets (Phase 3)
docker exec kafka kafka-topics --create `
    --bootstrap-server localhost:9092 `
    --topic apple_tweets `
    --partitions 1 `
    --replication-factor 1 `
    --if-not-exists

# Vérifier les topics créés
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
```

Résultat attendu : `apple_tweets` et `sentiment_stream` listés

---

## ÉTAPE 4 — Démarrer le conteneur spark-submit

Ce conteneur est le "lanceur" de tous vos jobs PySpark.
Il tourne en arrière-plan et vous exécutez vos scripts dedans.

```powershell
# Démarrer spark-submit (profile tools)
docker-compose --profile tools up -d spark-submit

# Vérifier qu'il est running
docker ps | findstr spark-submit
```

---

## ÉTAPE 5 — PHASE 1 : Entraînement du modèle

```powershell
# Lancer le preprocessing
docker exec spark-submit python src/preprocessing.py

# Lancer l'entraînement complet (1,6M tweets — ~10-15 min)
docker exec spark-submit python src/train_model.py

# Ou test rapide avec 10% du dataset (~1-2 min)
docker exec spark-submit python src/train_model.py --sample 0.1

# Vérifier que le modèle a bien été créé
docker exec spark-submit ls -la /workspace/data/models/sentiment_model/
```

Vous devriez voir le modèle dans `data\models\sentiment_model\` sur Windows aussi
(grâce au volume monté).

---

## ÉTAPE 6 — PHASE 2 : Big Data Streaming Sentiment140

### Terminal 1 — Lancer le consumer Spark Streaming

```powershell
# Dans un premier terminal PowerShell
docker exec spark-submit python src/spark_streaming_consumer_sentiment140.py
```

Attendez de voir : `Streaming démarré — micro-batch toutes les 10s`

### Terminal 2 — Lancer le producer Kafka (sur Windows)

```powershell
# Dans un deuxième terminal PowerShell
# Activer le venv
.\venv\Scripts\activate

# Lancer le producer (tourne sur Windows, envoie vers localhost:9092)
python src/kafka_producer_sentiment140.py

# Ou test rapide : seulement 50 000 tweets
python src/kafka_producer_sentiment140.py --max-rows 50000
```

### Vérifier dans Kafka UI

Ouvrir http://localhost:8080 → Topics → sentiment_stream → voir les messages entrants

### Vérifier les fichiers Parquet créés

```powershell
# Sur Windows, vérifier le dossier data
dir data\streaming\sentiment140_predictions\
```

---

## ÉTAPE 7 — PHASE 3 : Apple Brand Monitoring

### Terminal 3 — Consumer Apple Spark

```powershell
docker exec spark-submit python src/spark_streaming_consumer_apple.py
```

### Terminal 4 — Producer Apple (sur Windows)

```powershell
.\venv\Scripts\activate
python src/kafka_producer_apple.py

# Mode boucle pour démo continue
python src/kafka_producer_apple.py --loop
```

---

## ÉTAPE 8 — Dashboard Streamlit (sur Windows)

```powershell
# Dans un nouveau terminal PowerShell
.\venv\Scripts\activate
streamlit run dashboard/app.py
```

Accès : http://localhost:8501

> Le dashboard lit directement les fichiers Parquet dans `data\streaming\`
> qui sont écrits par Spark dans Docker (via le volume partagé).

---

## COMMANDES UTILES

### Voir les logs d'un conteneur

```powershell
# Logs Spark Master
docker logs spark-master --tail 50

# Logs du job PySpark en cours
docker logs spark-submit --tail 100 -f

# Logs Kafka
docker logs kafka --tail 30
```

### Entrer dans un conteneur pour déboguer

```powershell
# Shell interactif dans spark-submit
docker exec -it spark-submit bash

# Une fois dans le conteneur :
ls /workspace/data/
python -c "from config.config import *; print(SPARK_MASTER, KAFKA_BOOTSTRAP_SERVERS)"
```

### Redémarrer un service

```powershell
docker-compose restart spark-worker
docker-compose restart kafka
```

### Arrêter toute la stack

```powershell
# Arrêt propre (conserve les volumes)
docker-compose --profile tools down

# Arrêt + suppression des volumes (reset complet)
docker-compose --profile tools down -v
```

### Réinitialiser les checkpoints et prédictions

```powershell
# Sur Windows PowerShell
Remove-Item -Recurse -Force data\streaming\sentiment140_predictions
Remove-Item -Recurse -Force data\streaming\apple_predictions
Remove-Item -Recurse -Force checkpoint
New-Item -ItemType Directory data\streaming\sentiment140_predictions
New-Item -ItemType Directory data\streaming\apple_predictions
```

---

## DÉPANNAGE

### Problème : "Cannot connect to Kafka" dans Spark Streaming

**Cause** : Le consumer Spark dans Docker utilise `kafka:29092` mais
l'env var KAFKA_BOOTSTRAP_SERVERS n'est pas définie.

**Solution** : Vérifier que le conteneur spark-submit a bien l'env var :
```powershell
docker exec spark-submit env | findstr KAFKA
# Doit afficher : KAFKA_BOOTSTRAP_SERVERS=kafka:29092
```

### Problème : "No such file or directory: /workspace/data/raw/sentiment140.csv"

**Cause** : Le volume `./data` est mal monté.

**Solution** : Vérifier que les fichiers sont bien dans `data\raw\` sur Windows :
```powershell
dir data\raw\
# Doit afficher : sentiment140.csv  apple_tweets.csv
```

Puis vérifier le montage dans Docker :
```powershell
docker exec spark-submit ls /workspace/data/raw/
```

### Problème : "spark-submit container exited"

**Cause** : Le conteneur s'est arrêté avant votre commande.

**Solution** :
```powershell
docker-compose --profile tools up -d spark-submit
docker ps | findstr spark-submit
```

### Problème : Parquet non créé / vide

**Cause** : Permission sur le dossier `data/` ou le checkpoint est corrompu.

**Solution** :
```powershell
# Vérifier les permissions depuis Docker
docker exec spark-submit ls -la /workspace/data/streaming/

# Si le dossier n'existe pas, le créer depuis Docker
docker exec spark-submit mkdir -p /workspace/data/streaming/sentiment140_predictions
docker exec spark-submit mkdir -p /workspace/data/streaming/apple_predictions
docker exec spark-submit mkdir -p /workspace/checkpoint/sentiment140
docker exec spark-submit mkdir -p /workspace/checkpoint/apple
```

### Problème : "ImportError: No module named 'findspark'"

**Cause** : Vous avez encore `import findspark` dans un notebook ou script.

**Solution** : Supprimer toutes les occurrences de findspark (voir Étape 1 de ce guide).

---

## ORDRE D'EXÉCUTION COMPLET (récapitulatif)

```
PowerShell #1 (setup)
  docker-compose build
  docker-compose up -d
  docker-compose --profile tools up -d spark-submit
  docker exec kafka kafka-topics --create ... sentiment_stream ...
  docker exec kafka kafka-topics --create ... apple_tweets ...

PowerShell #1 (Phase 1 — preprocessing)
  docker exec spark-submit python src/preprocessing.py

PowerShell #1 (Phase 1)
  docker exec spark-submit python src/train_model.py

PowerShell #1 (Phase 2 — consumer)
  docker exec spark-submit python src/spark_streaming_consumer_sentiment140.py

PowerShell #2 (Phase 2 — producer Windows)
  .\venv\Scripts\activate
  python src/kafka_producer_sentiment140.py

PowerShell #3 (Phase 3 — consumer)
  docker exec spark-submit python src/spark_streaming_consumer_apple.py

PowerShell #4 (Phase 3 — producer Windows)
  .\venv\Scripts\activate
  python src/kafka_producer_apple.py --loop

PowerShell #5 (Dashboard)
  .\venv\Scripts\activate
  streamlit run dashboard/app.py
```
