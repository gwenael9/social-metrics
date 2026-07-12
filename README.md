# SocialMetrics AI — API d'Analyse de Sentiments

API Flask permettant d'évaluer le sentiment de tweets grâce à un modèle de
régression logistique entraîné sur des données annotées stockées en MySQL.

## Architecture

```
app/
  api.py              # application Flask, endpoint POST /api/sentiment
  db.py                # connexion MySQL
  sentiment_model.py   # modèle ML (TF-IDF + 2x LogisticRegression)
scripts/
  schema.sql           # création base + table tweets
  seed_db.py            # import du dataset d'exemple
  train.py              # entraînement initial
  retrain.py             # réentraînement (appelé par cron)
  evaluate.py            # matrices de confusion + précision/rappel/F1
cron/
  retrain.sh              # wrapper shell pour le cronjob
  crontab.txt              # exemple de configuration cron
data/
  seed_tweets.csv           # dataset d'exemple annoté (français)
model/                       # artefacts du modèle entraîné (généré)
report/                       # matrices de confusion + métriques (généré)
run.py                         # point d'entrée pour lancer l'API
```

## Installation

1. Cloner le repo et créer un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copier `.env.example` en `.env` et renseigner les identifiants MySQL :

```bash
cp .env.example .env
```

3. Créer la base et la table `tweets` :

```bash
mysql -u root -p < scripts/schema.sql
```

4. Charger le dataset d'exemple :

```bash
python scripts/seed_db.py
```

5. Entraîner le modèle une première fois :

```bash
python scripts/train.py
```

## Lancer l'API

```bash
python run.py
```

Par défaut l'API écoute sur `http://0.0.0.0:5000`.

## Utilisation de l'API

### `POST /api/sentiment`

Corps de la requête : soit un tableau de chaînes, soit un objet `{"tweets": [...]}`.

```bash
curl -X POST http://localhost:5000/api/sentiment \
  -H "Content-Type: application/json" \
  -d '["J'\''adore ce produit !", "Service catastrophique."]'
```

Réponse :

```json
{
  "J'adore ce produit !": 0.82,
  "Service catastrophique.": -0.76
}
```

Le score va de **-1** (très négatif) à **1** (très positif), calculé comme
`P(positif) - P(négatif)`.

Erreurs gérées :
- `400` : corps JSON absent/invalide, liste vide, éléments non-string.
- `503` : modèle non entraîné (lancer `scripts/train.py`).

### `GET /api/health`

Vérification simple de disponibilité, retourne `{"status": "ok"}`.

## Réentraînement du modèle

`scripts/retrain.py` recharge toutes les données de la table `tweets` et
réentraîne le modèle. À automatiser chaque semaine via cron :

```bash
crontab -e
# ajouter (voir cron/crontab.txt) :
0 3 * * 1 /chemin/vers/social-metrics/cron/retrain.sh
```

Les logs sont écrits dans `logs/retrain.log`.

## Évaluation du modèle

```bash
python scripts/evaluate.py
```

Génère :
- `report/confusion_positive.png`
- `report/confusion_negative.png`
- `report/metrics.txt` (précision, rappel, F1 pour chaque label)

Voir `report/rapport_evaluation.pdf` pour l'analyse détaillée des performances.
