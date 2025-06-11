import json
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login
#login(token=os.getenv("HF_TOKEN"))

# from accelerate import Accelerator # Décommenter si vous comptez l'utiliser explicitement, mais pas essentiel ici

# --- Configuration de Flask ---
app = Flask(__name__)

# --- Configuration du modèle DORA ---
MODEL_PATH = "/app/dora-model" #"xijins/DE50-project" # Assurez-vous que c'est bien le nom exact du modèle Hugging Face
MAX_NEW_TOKENS = 100 # Ajustez selon la longueur de réponse désirée et la performance (GPU devrait être rapide)

# Variables globales pour le modèle et le tokenizer
model = None
tokenizer = None
def load_dora_model():
    global model, tokenizer
    try:
        #print(f"[DORA] Authentification Hugging Face...")
        #login(token=os.getenv("HF_TOKEN"))  # <-- Ajout ici

        print(f"[DORA] Chargement du modèle depuis : {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        print("[DORA] Modèle chargé avec succès !")
    except Exception as e:
        print(f"[DORA] Erreur critique : {e}")
        model = None
        tokenizer = None

load_dora_model()

def repondre_avec_dora(question: str) -> str:
    """
    Génère une réponse avec le modèle DORA pour une question donnée.
    """
    if model is None or tokenizer is None:
        print("[DORA] Erreur: Modèle non chargé pour la génération.")
        return "Erreur interne du serveur : le modèle DORA n'est pas chargé."

    try:
        print(f"[DORA] --- Début de la génération pour : {question[:50]}... ---")
        inputs = tokenizer(question, return_tensors="pt")

        # Les inputs sont déjà sur le bon device grâce à device_map="auto" dans from_pretrained,
        # mais on peut s'assurer qu'ils sont sur le device du modèle si besoin.
        # inputs = {k: v.to(model.device) for k, v in inputs.items()} # Décommenter si problèmes

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if question in response:
            answer = response[len(question):].strip()
        else:
            answer = response.strip()

        print(f"[DORA] --- Fin de la génération. Réponse (début) : {answer[:50]}... ---")
        return answer

    except Exception as e:
        print(f"[DORA] !!! ERREUR DANS repondre_avec_dora pour '{question[:50]}...' : {e}")
        import traceback
        traceback.print_exc()
        return f"Désolé, une erreur est survenue lors de la génération de la réponse : {str(e)}"

# --- Endpoint API pour les développeurs mobiles ---
@app.route('/ask_dora', methods=['POST'])
def ask_dora():
    """
    Endpoint de l'API pour poser une question à DORA.
    Attend un JSON avec la clé 'question'.
    """
    print(f"[API] Requête reçue de : {request.remote_addr}")
    # print(f"[API] Headers : {request.headers}") # Moins utile en production
    # print(f"[API] Données brutes : {request.get_data()}") # Moins utile en production

    if not request.is_json:
        print("[API] Erreur: Content-Type non 'application/json'")
        return jsonify({"error": "Content-Type doit être 'application/json'"}), 400

    try:
        data = request.get_json()
    except Exception as e:
        print(f"[API] Erreur lors du décodage JSON : {e}")
        return jsonify({"error": f"JSON invalide : {str(e)}"}), 400

    question = data.get('question')

    if not question:
        print("[API] Erreur: Champ 'question' manquant.")
        return jsonify({"error": "Le champ 'question' est manquant dans la requête JSON"}), 400

    print(f"[API] Question reçue : {question}")
    answer = repondre_avec_dora(question)
    print(f"[API] Réponse générée : {answer[:100]}...")

    return jsonify({"question": question, "answer": answer})

# --- Point d'entrée de l'application ---
if __name__ == '__main__':
    print("[FLASK] Démarrage de l'application Flask...")
        #load_dora_model()
    # Cloud Run fournit le port via la variable d'environnement PORT (par défaut 8080)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False) # Désactivez le debug pour la production
