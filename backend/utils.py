# utils.py

import logging
import os

# ============================
# Initialisation du logger
# ============================

def setup_logger(log_file_path="logs/cbir.log", log_level="INFO"):
    """
    Configure le logger pour le projet.
    
    :param log_file_path: Le chemin du fichier où les logs seront enregistrés.
    :param log_level: Le niveau des logs (par défaut 'INFO').
    :return: L'instance du logger configuré.
    """
    # Crée le répertoire des logs s'il n'existe pas
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configuration du logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Création d'un handler pour écrire les logs dans un fichier
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    
    # Création d'un format de log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Ajout du handler au logger
    logger.addHandler(file_handler)
    
    # Ajout d'un handler pour afficher les logs dans la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# ============================
# Gestion des erreurs
# ============================

def handle_error(error_message):
    """
    Gère les erreurs en les loggant et en renvoyant une réponse appropriée.
    
    :param error_message: Message d'erreur à logger.
    """
    logger = logging.getLogger()
    logger.error(error_message)
    # Vous pouvez personnaliser la manière dont vous gérez les erreurs, par exemple,
    # en retournant un message spécifique ou en levant une exception.
    raise Exception(error_message)

# ============================
# Validation d'entrées
# ============================

def validate_image_file(image_path):
    """
    Valide si un fichier d'image existe et si son format est correct.
    
    :param image_path: Le chemin du fichier d'image.
    :return: True si l'image est valide, sinon False.
    """
    if not os.path.exists(image_path):
        handle_error(f"Le fichier {image_path} n'existe pas.")
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        handle_error(f"Le fichier {image_path} n'a pas une extension valide.")
    
    return True


