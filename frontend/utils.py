
import os
import io
import logging
from PIL import Image
import requests

# ============================
# Initialisation du logger
# ============================
logger = logging.getLogger("frontend")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# ============================
# Fonction pour valider le fichier image
# ============================
def validate_image_file(image_file):
    """
    Valide si le fichier est bien une image.
    Retourne True si l'image est valide, False sinon.
    """
    try:
        img = Image.open(image_file)
        img.verify()  # Vérifie que l'image est bien valide
        return True
    except (IOError, ValueError):
        logger.error(f"Le fichier {image_file} n'est pas une image valide.")
        return False

# ============================
# Fonction pour afficher les erreurs
# ============================
def show_error(message):
    """
    Affiche un message d'erreur dans l'interface utilisateur.
    """
    logger.error(message)
    return f"Erreur : {message}"

# ============================
# Fonction pour télécharger l'image
# ============================
def upload_image(file):
    """
    Envoie l'image vers l'API backend pour indexation.
    """
    try:
        response = requests.post(
            "http://localhost:8000/index_images/", 
            files={"file": file}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return show_error(f"Erreur lors de l'indexation de l'image: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de connexion : {str(e)}")
        return show_error("Erreur de connexion au backend.")
