import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import joblib
import time

SR = 16000
DUREE = 3  # secondes
FICHIER_TEMP = "audio_test.wav"

# Charger les modèles
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
clf = joblib.load("modele_pleurs (2).joblib")

def enregistrer_audio(fichier, duree=DUREE, sr=SR):
    print("🎙️ Enregistrement...")
    audio = sd.rec(int(duree * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    sf.write(fichier, audio, sr)
    return audio.flatten()

def extraire_embedding_yamnet(signal):
    waveform = tf.convert_to_tensor(signal, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)

def predire(embedding):
    score = clf.predict_proba(embedding)[0][1]  # proba de la classe "pleurs"
    prediction = "PLEURS" if score > 0.75 else "AUTRE"
    return prediction, score


print("🧠 Détecteur en temps réel avec YAMNet + RandomForest")
print("🔁 Appuie sur Ctrl+C pour arrêter\n")

try:
    while True:
        audio_signal = enregistrer_audio(FICHIER_TEMP)
        vecteur = extraire_embedding_yamnet(audio_signal)
        resultat, proba = predire(vecteur)
        print(f"🧠 {resultat.upper()} ({proba:.2f})\n{'-'*30}")
        time.sleep(3)

except KeyboardInterrupt:
    print("\n🛑 Fin de la détection.")
