"""Entraîne le modèle de sentiment sur tous les tweets présents en base."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import train_test_split

from app.db import fetch_all_tweets
from app.sentiment_model import SentimentModel


def load_dataset():
    rows = fetch_all_tweets()
    if not rows:
        raise RuntimeError("Aucun tweet en base. Lancez d'abord scripts/seed_db.py.")
    texts = [r["text"] for r in rows]
    positive = [r["positive"] for r in rows]
    negative = [r["negative"] for r in rows]
    return texts, positive, negative


def train():
    texts, positive, negative = load_dataset()

    X_train, X_val, y_pos_train, y_pos_val, y_neg_train, y_neg_val = train_test_split(
        texts, positive, negative, test_size=0.2, random_state=42
    )

    model = SentimentModel()
    model.fit(X_train, y_pos_train, y_neg_train)
    model.save()

    print(f"Entraîné sur {len(X_train)} tweets, {len(X_val)} mis de côté pour validation.")
    print("Modèle sauvegardé dans", os.getenv("MODEL_DIR", "model"))

    return model, X_val, y_pos_val, y_neg_val


if __name__ == "__main__":
    train()
