import requests
import urllib.parse

# ✅ Remplace ces valeurs par les tiennes
numero = "+24371363184"              # Ton numéro WhatsApp
cle_api = "8393346"              # Clé API reçue par CallMeBot
message = "🍼 Alerte TEST : pleurs détectés !"

# Encoder proprement le message pour l'URL
texte_encode = urllib.parse.quote(message)
url = f"https://api.callmebot.com/whatsapp.php?phone=+243971363184&text=BONJOUR&apikey=8393346"

try:
    reponse = requests.get(url)
    if "Message successfully sent" in reponse.text:
        print("✅ Message WhatsApp envoyé avec succès !")
    else:
        print("❌ Réponse inattendue :", reponse.text)
except Exception as e:
    print("❌ Erreur lors de l’envoi :", e)
