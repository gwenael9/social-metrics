"""Évalue le modèle : matrices de confusion + précision/rappel/F1
pour les labels positive et negative, sur un jeu de validation.
Génère report/confusion_positive.png, report/confusion_negative.png
et report/metrics.txt.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db import fetch_all_tweets
from app.sentiment_model import SentimentModel

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")


def load_split():
    rows = fetch_all_tweets()
    if not rows:
        raise RuntimeError("Aucun tweet en base. Lancez d'abord scripts/seed_db.py.")
    texts = [r["text"] for r in rows]
    positive = [r["positive"] for r in rows]
    negative = [r["negative"] for r in rows]
    return train_test_split(texts, positive, negative, test_size=0.2, random_state=42)


def plot_confusion(cm, labels, title, path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate():
    X_train, X_val, y_pos_train, y_pos_val, y_neg_train, y_neg_val = load_split()

    model = SentimentModel()
    model.fit(X_train, y_pos_train, y_neg_train)

    pred_pos, pred_neg = model.predict_labels(X_val)

    os.makedirs(REPORT_DIR, exist_ok=True)

    cm_pos = confusion_matrix(y_pos_val, pred_pos, labels=[0, 1])
    cm_neg = confusion_matrix(y_neg_val, pred_neg, labels=[0, 1])

    plot_confusion(
        cm_pos, ["non-positif", "positif"], "Matrice de confusion - positive",
        os.path.join(REPORT_DIR, "confusion_positive.png"),
    )
    plot_confusion(
        cm_neg, ["non-négatif", "négatif"], "Matrice de confusion - negative",
        os.path.join(REPORT_DIR, "confusion_negative.png"),
    )

    report_pos = classification_report(y_pos_val, pred_pos, target_names=["non-positif", "positif"], zero_division=0)
    report_neg = classification_report(y_neg_val, pred_neg, target_names=["non-négatif", "négatif"], zero_division=0)

    texte = []
    texte.append(f"Jeu de validation : {len(X_val)} tweets (sur {len(X_train) + len(X_val)} au total)\n")
    texte.append("=== Label positive ===\n")
    texte.append(f"Matrice de confusion :\n{cm_pos}\n")
    texte.append(report_pos + "\n")
    texte.append("=== Label negative ===\n")
    texte.append(f"Matrice de confusion :\n{cm_neg}\n")
    texte.append(report_neg + "\n")

    rapport = "\n".join(texte)
    print(rapport)

    with open(os.path.join(REPORT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(rapport)

    print(f"Matrices de confusion et rapport sauvegardés dans {REPORT_DIR}/")

    return {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "cm_pos": cm_pos,
        "cm_neg": cm_neg,
        "report_pos": report_pos,
        "report_neg": report_neg,
        "img_pos": os.path.join(REPORT_DIR, "confusion_positive.png"),
        "img_neg": os.path.join(REPORT_DIR, "confusion_negative.png"),
    }


if __name__ == "__main__":
    evaluate()
