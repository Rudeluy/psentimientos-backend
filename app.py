from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("sentiment_model.pkl")

@app.route("/")
def home():
    return "API de Análisis de Sentimientos activa"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "Debe ingresar un texto"}), 400

        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0][int(pred)]

        sentimiento = "positivo" if pred == 1 else "negativo"

        return jsonify({
            "sentimiento": sentimiento,
            "probabilidad": round(float(prob), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    from os import environ
    app.run(host="0.0.0.0", port=int(environ.get("PORT", 5000)))
