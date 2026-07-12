import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_DIR = os.getenv("MODEL_DIR", "model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
CLF_POS_PATH = os.path.join(MODEL_DIR, "clf_pos.joblib")
CLF_NEG_PATH = os.path.join(MODEL_DIR, "clf_neg.joblib")


class SentimentModel:
    """Score = P(positif) - P(négatif), calculé à partir de deux LogisticRegression
    (une sur le label `positive`, une sur le label `negative`) partageant le même
    vectoriseur TF-IDF.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.clf_pos = LogisticRegression(max_iter=1000)
        self.clf_neg = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(self, texts, positive_labels, negative_labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf_pos.fit(X, positive_labels)
        self.clf_neg.fit(X, negative_labels)
        self._fitted = True

    def predict_scores(self, texts):
        if not self._fitted:
            raise RuntimeError("Modèle non entraîné ou non chargé.")
        X = self.vectorizer.transform(texts)
        p_pos = self.clf_pos.predict_proba(X)[:, 1]
        p_neg = self.clf_neg.predict_proba(X)[:, 1]
        scores = np.clip(p_pos - p_neg, -1.0, 1.0)
        return scores.tolist()

    def predict_labels(self, texts):
        if not self._fitted:
            raise RuntimeError("Modèle non entraîné ou non chargé.")
        X = self.vectorizer.transform(texts)
        return self.clf_pos.predict(X), self.clf_neg.predict(X)

    def save(self, model_dir=None):
        model_dir = model_dir or MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(model_dir, "vectorizer.joblib"))
        joblib.dump(self.clf_pos, os.path.join(model_dir, "clf_pos.joblib"))
        joblib.dump(self.clf_neg, os.path.join(model_dir, "clf_neg.joblib"))

    @classmethod
    def load(cls, model_dir=None):
        model_dir = model_dir or MODEL_DIR
        instance = cls()
        instance.vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
        instance.clf_pos = joblib.load(os.path.join(model_dir, "clf_pos.joblib"))
        instance.clf_neg = joblib.load(os.path.join(model_dir, "clf_neg.joblib"))
        instance._fitted = True
        return instance
