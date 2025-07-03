from flask import Flask, request, jsonify, render_template, send_file
import soundfile as sf
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import joblib
import requests
import urllib.parse
import time
import qrcode
import logging
from io import BytesIO
import socket
import os

app = Flask(__name__)

# 🔧 Chargement des modèles
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
rf_model = joblib.load("modele_random_forest.joblib")
mlp_model = tf.keras.models.load_model("modele_mlp.h5")
cnn_model = tf.keras.models.load_model("modele_cnn1d.h5")

# 🔐 Configuration API WhatsApp
API_KEY = "8393346"
dernier_envoi = 0
delai_alerte = 180  # secondes

# 🧾 Configuration des logs
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def envoyer_whatsapp(message, numero):
    global dernier_envoi
    maintenant = time.time()
    if maintenant - dernier_envoi < delai_alerte:
        logging.info("⏳ Alerte ignorée (intervalle trop court)")
        return
    url = f"https://api.callmebot.com/whatsapp.php?phone={numero}&text={urllib.parse.quote(message)}&apikey={API_KEY}"
    try:
        res = requests.get(url)
        if "Message successfully sent" in res.text:
            logging.info("✅ Alerte envoyée à %s", numero)
            dernier_envoi = maintenant
        else:
            logging.error("❌ Erreur d'envoi : %s", res.text)
    except Exception as e:
        logging.exception("❌ Exception lors de l'envoi WhatsApp")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stream", methods=["POST"])
def stream():
    fichier = request.files.get("audio")
    numero = request.form.get("numero", "").strip()
    modele = request.form.get("modele", "rf").strip().lower()

    if not fichier or not numero:
        return jsonify({"prediction": "NON VALIDE", "score": 0.0})

    fichier.save("segment.wav")
    try:
        audio, sr = sf.read("segment.wav")
        if audio.ndim > 1:
            audio = audio[:, 0]  # mono
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if len(audio) < 16000:
            raise ValueError("Signal audio trop court (moins d'une seconde)")

        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        _, embeddings, _ = yamnet(waveform)

        if embeddings.shape[0] < 3:
            raise ValueError("Pas assez de frames YamNet")

        vecteur = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)
        sequence = embeddings.numpy()[np.newaxis, :, :]
        n_frames = sequence.shape[1]

        if modele == "rf":
            score = float(rf_model.predict_proba(vecteur)[0][1])

        elif modele == "mlp":
            score = float(mlp_model.predict(vecteur)[0][0])

        elif modele == "cnn":
            if n_frames < 3:
                raise ValueError("CNN nécessite ≥3 frames")
            score = float(cnn_model.predict(sequence)[0][0])

        elif modele == "fusion":
            if n_frames < 3:
                raise ValueError("Fusion nécessite ≥3 frames")
            score_cnn = float(cnn_model.predict(sequence)[0][0])
            score_mlp = float(mlp_model.predict(vecteur)[0][0])
            score = 0.5 * score_cnn + 0.5 * score_mlp

        else:
            raise ValueError("Modèle inconnu")

        prediction = "PLEURS" if score > 0.90 else "AUTRE"
        logging.info(f"📊 Modèle: {modele} | Prédiction: {prediction} | Score: {score:.3f}")

        if prediction == "PLEURS":
            envoyer_whatsapp("🍼 Alerte : pleurs détectés !", numero)

        return jsonify({"prediction": prediction, "score": round(score, 3)})

    except Exception as e:
        logging.exception("⚠️ Erreur traitement audio")
        return jsonify({"prediction": "NON VALIDE", "score": 0.0})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)