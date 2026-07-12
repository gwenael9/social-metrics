import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from app.sentiment_model import SentimentModel

load_dotenv()

app = Flask(__name__)

_model = None


def get_model():
    """Charge le modèle en mémoire au premier appel (singleton)."""
    global _model
    if _model is None:
        model_dir = os.getenv("MODEL_DIR", "model")
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                "Aucun modèle entraîné trouvé. Lancez scripts/train.py avant de démarrer l'API."
            )
        _model = SentimentModel.load(model_dir)
    return _model


def extract_tweets(payload):
    """Extrait la liste de tweets du corps JSON, qui peut être soit un
    tableau de chaînes directement, soit un objet {"tweets": [...]}.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "tweets" in payload:
        return payload["tweets"]
    return None


@app.route("/api/sentiment", methods=["POST"])
def analyze_sentiment():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Corps JSON invalide ou absent."}), 400

    tweets = extract_tweets(payload)
    if tweets is None:
        return jsonify({
            "error": "Format invalide. Attendu : un tableau de chaînes (string[]) "
                     "ou un objet {\"tweets\": [...]}."
        }), 400

    if not isinstance(tweets, list) or len(tweets) == 0:
        return jsonify({"error": "La liste de tweets ne peut pas être vide."}), 400

    if not all(isinstance(t, str) and t.strip() for t in tweets):
        return jsonify({"error": "Chaque tweet doit être une chaîne de caractères non vide."}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    scores = model.predict_scores(tweets)
    result = {tweet: round(float(score), 4) for tweet, score in zip(tweets, scores)}
    return jsonify(result), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
    )
