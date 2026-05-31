# 📊 Real-Time Sentiment Analysis — Big Data Pipeline

> **Kafka · Spark Structured Streaming · MLlib · Parquet · Streamlit**  
> Surveillance de réputation de marque en temps réel — Projet Big Data 2024-2025

---

## 📋 Table des matières

1. [Problématique métier](#-problématique-métier)
2. [Architecture en trois phases](#-architecture-en-trois-phases)
3. [Datasets](#-datasets)
4. [Technologies](#-technologies)
5. [Big Data Value Chain](#-big-data-value-chain)
6. [Installation](#-installation-rapide)
7. [Exécution](#-exécution)
8. [Structure du projet](#-structure-du-projet)
9. [Dashboard](#-dashboard-streamlit)
10. [Résultats attendus](#-résultats-attendus)
11. [Répartition du travail](#-répartition-du-travail)
12. [Limites et perspectives](#-limites-et-perspectives)

---

## 🎯 Problématique métier

> *Comment concevoir un pipeline Big Data complet capable (1) d'apprendre les patterns de sentiment à partir d'un corpus massif de tweets, (2) de traiter ce corpus en streaming temps réel à grande échelle via Kafka et Spark, et (3) de déployer ce système pour surveiller la réputation d'une marque spécifique ?*

Dans un environnement numérique où des millions de messages sont publiés chaque heure sur les réseaux sociaux, les entreprises doivent analyser automatiquement les sentiments exprimés dans des flux massifs et détecter en temps réel toute dégradation de leur image de marque.

---

## 🏗️ Architecture en trois phases

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Offline Batch Training                                        │
│  Sentiment140 CSV → Spark Batch → NLP (TF-IDF) → Logistic Regression   │
│                                              ↓                           │
│                                    Modèle ML sauvegardé (Parquet)       │
└─────────────────────────────────────────────────────────────────────────┘
          │ Modèle chargé                    │ Modèle chargé
          ↓                                  ↓
┌────────────────────────┐    ┌────────────────────────────────────────────┐
│  PHASE 2               │    │  PHASE 3                                   │
│  Big Data Streaming    │    │  Brand Monitoring                          │
│                        │    │                                            │
│  Sentiment140          │    │  Apple Tweets CSV                          │
│  ↓ Kafka Producer      │    │  ↓ Kafka Producer                          │
│  ↓ Kafka Topic         │    │  ↓ Kafka Topic (apple_tweets)             │
│  ↓ Spark Streaming     │    │  ↓ Spark Streaming                         │
│  ↓ ML Inference        │    │  ↓ ML Inference                            │
│  ↓ Neutral Layer       │    │  ↓ Neutral Layer                           │
│  ↓ Parquet Data Lake   │    │  ↓ Streamlit Dashboard                     │
│    1,6M prédictions    │    │    Bad Buzz Alerts                         │
└────────────────────────┘    └────────────────────────────────────────────┘
```

### Pourquoi Sentiment140 en Phase 2 ?

Le dataset Sentiment140 (1,6M tweets) est utilisé à la fois en Phase 1 (entraînement) **et** en Phase 2 (streaming) pour deux raisons fondamentales :

1. **Volume Big Data** : 1,6M tweets démontrent la capacité de traitement à grande échelle, contrairement aux 9K tweets Apple
2. **Conformité** : Les exigences du projet imposent ≥ 1M lignes dans le pipeline principal

Le dataset Apple (~9K tweets) est le **cas d'usage métier final**, pas le pipeline principal.

---

## 📂 Datasets

### Dataset 1 — Sentiment140 (Phases 1 et 2)

| Caractéristique | Valeur |
|----------------|--------|
| Volume | **1 600 000 tweets** |
| Colonnes | 6 : `target, id, date, flag, user, text` |
| Taille | ~230 MB (CSV brut) |
| Source | API Twitter officielle (Stanford) |
| Usage ML | Colonnes `text` (feature) + `target` (label) uniquement |
| Classes | 800K négatifs (target=0) + 800K positifs (target=4) |
| Téléchargement | http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip |

### Dataset 2 — Apple Tweets (Phase 3)

| Caractéristique | Valeur |
|----------------|--------|
| Volume | ~9 000 tweets |
| Colonnes | `tweet_text, emotion_in_tweet_is_directed_at, is_there_an_emotion_...` |
| Usage | Brand Monitoring Apple (iPhone, iPad, iOS, MacBook) |
| Source | Kaggle — Apple Twitter Sentiment Dataset |

---

## 🛠️ Technologies

| Technologie | Version | Rôle |
|------------|---------|------|
| **Apache Kafka** | 3.7 | Ingestion temps réel, messaging distribué |
| **Apache Spark** | 3.5.1 | Batch + Structured Streaming |
| **Spark MLlib** | 3.5.1 | Logistic Regression, TF-IDF pipeline |
| **Apache Parquet** | — | Data Lake colonnaire (modèle + prédictions) |
| **Streamlit** | 1.34 | Dashboard Brand Monitoring |
| **Plotly** | 5.22 | Visualisations interactives |
| **Python** | 3.10 | Langage principal |

---

## 🔗 Big Data Value Chain

| Étape | Phase | Technologie | Réalisation |
|-------|-------|------------|-------------|
| **Data Ingestion** | 1 | CSV Reader | Chargement 1,6M tweets |
| **Data Preprocessing** | 1 | PySpark NLP | Nettoyage + TF-IDF |
| **Data Storage** | 1,2,3 | Apache Parquet | Modèle + prédictions en Data Lake |
| **Data Processing** | 2,3 | Spark Streaming | Micro-batches 10s, 1,6M tweets |
| **Data Ingestion (RT)** | 2,3 | Apache Kafka | Flux temps réel |
| **Data Analysis + ML** | 1,2,3 | MLlib + UDF | Logistic Regression + Neutral Layer |
| **Data Visualization** | 3 | Streamlit | Dashboard + Bad Buzz Detection |

---

## ⚡ Installation rapide

### Option A — Docker (recommandé pour Kafka)

```bash
# Kafka + ZooKeeper + Interface UI
docker-compose up -d

# Vérification
docker-compose ps
```

### Option B — Installation manuelle

```bash
# Voir le guide complet : docs/INSTALLATION_GUIDE.sh

# 1. Java 11
sudo apt install -y openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# 2. Spark 3.5
wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
tar -xzf spark-3.5.1-bin-hadoop3.tgz && sudo mv spark-3.5.1-bin-hadoop3 /opt/spark

# 3. Kafka 3.7
wget https://downloads.apache.org/kafka/3.7.0/kafka_2.13-3.7.0.tgz
tar -xzf kafka_2.13-3.7.0.tgz && sudo mv kafka_2.13-3.7.0 /opt/kafka
```

### Environnement Python

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Exécution

### Préparer les topics Kafka

```bash



/opt/kafka/bin/kafka-topics.sh --create \
    --bootstrap-server localhost:9092 \
    --topic apple_tweets --partitions 1 --replication-factor 1
```

### Phase 1 — Entraînement du modèle

```bash
# Entraînement complet (1,6M tweets — ~10 min)
python src/train_model.py

# Test rapide (160K tweets — ~1 min)
python src/train_model.py --sample 0.1
```

### Phase 2 — Big Data Streaming (Sentiment140)

```bash
# Terminal 1 : Consumer Spark (en premier)
python src/spark_streaming_consumer_sentiment140.py

# Terminal 2 : Producer Kafka
python src/kafka_producer_sentiment140.py
# → 1,6M tweets en ~27 minutes à 1000 tweets/s
```

### Phase 3 — Apple Brand Monitoring

```bash
# Terminal 3 : Consumer Apple
python src/spark_streaming_consumer_apple.py

# Terminal 4 : Producer Apple (avec boucle pour démo continue)
python src/kafka_producer_apple.py --loop
```

### Dashboard Streamlit

```bash
streamlit run dashboard/app.py
# Accès : http://localhost:8501
```

---

## 📁 Structure du projet

```
sentiment-pipeline/
│
├── 📓 notebooks/
│   ├── 01_exploration_sentiment140.ipynb        # EDA + validation exigences
│   ├── 02_preprocessing_sentiment140_pyspark.ipynb
│   ├── 03_training_model_spark_mllib.ipynb      # Phase 1
│   ├── 04_model_evaluation_sentiment140.ipynb   # Métriques + confusion matrix
│   ├── 05_streaming_sentiment140_kafka_spark.ipynb  # Phase 2
│   ├── 06_apple_brand_monitoring_streaming.ipynb    # Phase 3
│   └── 07_dashboard_data_preparation.ipynb
│
├── 🐍 src/
│   ├── utils.py                           # Utilitaires partagés
│   ├── preprocessing.py                   # NLP Pipeline batch
│   ├── train_model.py                     # Phase 1 — Entraînement
│   ├── evaluate_model.py                  # Évaluation approfondie
│   ├── kafka_producer_sentiment140.py     # Phase 2 — Producer
│   ├── kafka_producer_apple.py            # Phase 3 — Producer
│   ├── spark_streaming_consumer_sentiment140.py  # Phase 2 — Consumer
│   └── spark_streaming_consumer_apple.py        # Phase 3 — Consumer
│
├── 📊 dashboard/
│   └── app.py                            # Dashboard Streamlit complet
│
├── ⚙️ config/
│   └── config.py                         # Configuration centralisée
│
├── 💾 data/
│   ├── raw/                              # CSV bruts (non versionnés)
│   ├── processed/                        # Parquet nettoyé
│   ├── models/sentiment_model/           # Modèle MLlib sérialisé
│   ├── streaming/
│   │   ├── sentiment140_predictions/     # Prédictions Phase 2
│   │   └── apple_predictions/           # Prédictions Phase 3
│   └── metrics/                          # Métriques d'évaluation
│
├── 📖 docs/
│   └── INSTALLATION_GUIDE.sh            # Guide d'installation complet
│
├── docker-compose.yml                    # Kafka + ZooKeeper + UI
├── requirements.txt                      # Dépendances Python
├── .gitignore                            # Exclusions Git
└── README.md                             # Ce fichier
```

---

## 📈 Dashboard Streamlit

### Fonctionnalités

| Composant | Description |
|-----------|-------------|
| **KPIs** | Total tweets, % Positifs, % Négatifs, % Neutres, Score réputation |
| **Statut marque** | 🟢 STABLE / 🟡 SURVEILLANCE / 🔴 BAD BUZZ |
| **Évolution temporelle** | Courbe empilée par minute (Pos/Neg/Neutral) |
| **Jauge réputation** | Score [0-100] basé sur Pos/(Pos+Neg) |
| **Camembert** | Distribution des 3 états de sentiment |
| **Distribution produit** | Histogramme groupé par produit Apple |
| **Top tweets négatifs** | Tableau des 10 tweets les plus négatifs |
| **Filtre source** | Switch Apple Brand Monitoring ↔ Sentiment140 |

### Logique métier — Couche Neutral

```
confidence = max(P(positif), P(négatif))

Si confidence < 0.60 → Neutral
Si confidence ≥ 0.60 et label=1 → Positive
Si confidence ≥ 0.60 et label=0 → Negative
```

### Détection de Bad Buzz

```
neg_ratio = N_Négatifs / (N_Positifs + N_Négatifs)   [hors Neutres]

neg_ratio > 70% sur fenêtre de 5 min → 🚨 ALERTE BAD BUZZ
```

---

## 🎯 Résultats attendus

| Métrique | Cible | Signification |
|----------|-------|--------------|
| Accuracy | ≥ 78% | Part des prédictions correctes |
| Precision | ≥ 76% | Pertinence des prédictions |
| Recall | ≥ 76% | Taux de détection |
| F1-score | ≥ 0.77 | Équilibre précision/rappel |
| AUC-ROC | ≥ 0.85 | Qualité de discrimination |
| Débit streaming | ~1 000 tweets/s | Sentiment140 via Kafka |
| Latence | < 15 s/tweet | Micro-batch de 10s |
| Volume traité | 1,6M tweets | Durée ~27 min |

---

## 👥 Répartition du travail en binôme

### Membre 1 — Infrastructure Big Data & Visualisation

| Tâche | Description |
|-------|-------------|
| Apache Kafka | Configuration, topics, producteurs |
| Spark Streaming | Consommateurs Phase 2 et Phase 3 |
| Couche Neutral | UDF Python, logique métier |
| Dashboard Streamlit | Interface, KPIs, alertes |
| Docker Compose | Infrastructure Kafka/ZooKeeper |
| Notebook 05 | Streaming Sentiment140 |
| Notebook 06 | Apple Brand Monitoring |

### Membre 2 — Machine Learning & Data Engineering

| Tâche | Description |
|-------|-------------|
| Preprocessing NLP | Pipeline nettoyage textuel |
| Feature Engineering | TF-IDF avec Spark MLlib |
| Entraînement ML | Logistic Regression, tuning |
| Évaluation modèle | Métriques, matrice de confusion |
| Data Lake | Structure Parquet, schémas |
| Documentation | Guide technique, docstrings |
| Notebook 01-04 | Exploration, preprocessing, training |

### Tâches communes

- Intégration des phases
- Tests end-to-end
- Rapport LaTeX
- Préparation de la soutenance
- README

---

## ⚠️ Limites et perspectives

### Limites actuelles

- Le modèle ML ne capture pas le sarcasme ni les emojis complexes
- Pas de gestion des langues autres que l'anglais
- Pas de fine-tuning spécifique sur les tweets Apple
- Latence minimale de 10s (micro-batch Spark)

### Perspectives d'amélioration

- Intégration d'un modèle Transformer (BERT) pour de meilleures performances
- Support multi-langues avec détection automatique
- API REST pour intégration avec d'autres systèmes
- Alerting Email/Slack en cas de Bad Buzz détecté
- Fine-tuning du modèle sur des tweets Apple spécifiques
- Passage à un cluster Kafka distribué pour la production

---

## 📚 Références

- [Sentiment140 — Stanford NLP](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Spark Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
- [Spark MLlib](https://spark.apache.org/docs/latest/ml-guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

*Projet Big Data — Modern Data Stack — Année universitaire 2024-2025*
