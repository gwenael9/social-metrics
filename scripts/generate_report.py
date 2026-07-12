"""Génère report/rapport_evaluation.pdf : matrices de confusion,
métriques (précision/rappel/F1), analyse des biais et recommandations.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.evaluate import evaluate

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")
PDF_PATH = os.path.join(REPORT_DIR, "rapport_evaluation.pdf")


def page_titre(pdf, resultats):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.5, 0.85, "SocialMetrics AI", ha="center", fontsize=24, weight="bold")
    fig.text(0.5, 0.80, "Rapport d'Évaluation du Modèle de Sentiment", ha="center", fontsize=16)
    fig.text(0.5, 0.75, "Client : Daunale Treupe", ha="center", fontsize=11)

    corps = (
        f"Ce rapport présente l'évaluation du modèle de régression logistique "
        f"utilisé par l'API d'analyse de sentiments.\n\n"
        f"Le modèle est composé de deux classifieurs LogisticRegression indépendants "
        f"(un pour le label \"positive\", un pour le label \"negative\") partageant un "
        f"même vectoriseur TF-IDF (unigrammes + bigrammes, 5000 features max).\n\n"
        f"Jeu de données : {resultats['n_train'] + resultats['n_val']} tweets annotés au total,\n"
        f"dont {resultats['n_train']} utilisés pour l'entraînement et "
        f"{resultats['n_val']} pour la validation (split 80/20, random_state=42)."
    )
    fig.text(0.1, 0.45, corps, fontsize=10.5, va="top", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def page_matrice(pdf, titre, img_path, interpretation):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.95, titre, ha="center", fontsize=15, weight="bold")

    img = mpimg.imread(img_path)
    ax_img = fig.add_axes([0.25, 0.55, 0.5, 0.35])
    ax_img.imshow(img)
    ax_img.axis("off")

    fig.text(0.1, 0.48, "Interprétation :", fontsize=11, weight="bold")
    fig.text(0.1, 0.44, interpretation, fontsize=10, va="top", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def page_metriques(pdf, resultats):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.95, "Précision, Rappel, F1-score", ha="center", fontsize=15, weight="bold")

    fig.text(0.1, 0.88, "Label positive :", fontsize=11, weight="bold")
    fig.text(0.1, 0.85, resultats["report_pos"], fontsize=8.5, va="top", family="monospace")

    fig.text(0.1, 0.55, "Label negative :", fontsize=11, weight="bold")
    fig.text(0.1, 0.52, resultats["report_neg"], fontsize=8.5, va="top", family="monospace")

    pdf.savefig(fig)
    plt.close(fig)


def page_analyse(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.95, "Analyse, Biais et Recommandations", ha="center", fontsize=15, weight="bold")

    texte = (
        "Observations sur les performances\n"
        "----------------------------------\n"
        "Le classifieur \"negative\" obtient des résultats corrects sur le jeu de\n"
        "validation, avec un F1-score raisonnable pour la classe négative. Le\n"
        "classifieur \"positive\" en revanche échoue à détecter la classe positive\n"
        "(rappel proche de 0) : il prédit quasi systématiquement la classe\n"
        "majoritaire \"non-positif\".\n\n"

        "Biais identifiés\n"
        "----------------\n"
        "1. Taille du dataset : seulement ~72 tweets annotés, dont 15 en\n"
        "   validation. Un jeu aussi réduit ne permet pas au modèle de\n"
        "   généraliser, et la performance mesurée est très sensible au split\n"
        "   train/validation.\n"
        "2. Déséquilibre de classes : sur un petit échantillon, la proportion\n"
        "   positif/négatif/neutre dans le split de validation varie fortement,\n"
        "   ce qui pénalise particulièrement la classe positive ici.\n"
        "3. Vectorisation TF-IDF avec bigrammes sur un vocabulaire restreint :\n"
        "   risque de surapprentissage (mots vus uniquement dans un contexte\n"
        "   précis à l'entraînement).\n"
        "4. Sensibilité aux accents/orthographe : le TF-IDF ne normalise pas les\n"
        "   variantes sans accents (\"genial\" vs \"génial\"), ce qui peut créer des\n"
        "   tokens hors-vocabulaire au moment de la prédiction.\n\n"

        "Recommandations\n"
        "----------------\n"
        "1. Collecter un dataset annoté beaucoup plus large (plusieurs milliers\n"
        "   de tweets) et représentatif du domaine réel (X / Twitter).\n"
        "2. Utiliser class_weight=\"balanced\" dans LogisticRegression pour\n"
        "   compenser le déséquilibre des classes.\n"
        "3. Valider avec une validation croisée (k-fold) plutôt qu'un split\n"
        "   unique, pour obtenir des métriques plus stables.\n"
        "4. Normaliser le texte (minuscules, suppression des accents,\n"
        "   lemmatisation) avant la vectorisation.\n"
        "5. Tester des modèles alternatifs ou complémentaires (embeddings\n"
        "   pré-entraînés, modèles de langue) et comparer via validation croisée.\n"
        "6. Réévaluer régulièrement (à chaque réentraînement hebdomadaire) pour\n"
        "   suivre la dérive des performances dans le temps."
    )
    fig.text(0.08, 0.88, texte, fontsize=9.5, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    resultats = evaluate()

    os.makedirs(REPORT_DIR, exist_ok=True)
    with PdfPages(PDF_PATH) as pdf:
        page_titre(pdf, resultats)
        page_matrice(
            pdf, "Matrice de confusion - Label positive", resultats["img_pos"],
            "Le modèle peine à identifier correctement les tweets positifs sur ce\n"
            "jeu de validation réduit : le rappel de la classe \"positif\" est très\n"
            "faible, la majorité des tweets positifs sont classés à tort comme\n"
            "\"non-positif\". Voir section Analyse pour les causes probables.",
        )
        page_matrice(
            pdf, "Matrice de confusion - Label negative", resultats["img_neg"],
            "Le modèle identifie correctement la majorité des tweets \"non-négatif\"\n"
            "et obtient un résultat correct sur la classe \"négatif\", bien que\n"
            "perfectible. Les performances restent limitées par la petite taille du\n"
            "jeu de données.",
        )
        page_metriques(pdf, resultats)
        page_analyse(pdf)

    print(f"Rapport PDF généré : {PDF_PATH}")


if __name__ == "__main__":
    main()
