"""Réentraîne le modèle avec les données les plus récentes de la table tweets.
Destiné à être appelé chaque semaine via cron (voir cron/retrain.sh).
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train import train


def main():
    horodatage = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[{horodatage}] Début du réentraînement hebdomadaire.")
    try:
        train()
    except RuntimeError as e:
        print(f"[{horodatage}] Réentraînement annulé : {e}")
        sys.exit(1)
    print(f"[{horodatage}] Réentraînement terminé avec succès.")


if __name__ == "__main__":
    main()
