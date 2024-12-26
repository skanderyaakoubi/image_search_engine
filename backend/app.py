# backend/app.py

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List
import os
import json
import logging

from .image_indexer_manager import ImageIndexerManager
from .feature_extractor import FeatureExtractor
from .utils import setup_logger, validate_image_file, handle_error
from .config import API_HOST, API_PORT, LOG_FILE_PATH, BATCH_SIZE

# ============================
# Initialisation du logger
# ============================
logger = setup_logger(log_file_path=LOG_FILE_PATH)

# ============================
# Initialisation de l'API FastAPI
# ============================
app = FastAPI()

# ============================
# Modèles Pydantic pour les réponses
# ============================

class SearchQuery(BaseModel):
    query_image: str  # Chemin de l'image de requête
    top_k: int = 5  # Nombre d'images à retourner (par défaut 5)

class IndexingResult(BaseModel):
    status: str
    message: str

class ImageMetadata(BaseModel):
    image_id: str
    features: List[float]  # Liste des caractéristiques extraites de l'image

# ============================
# Point de terminaison pour l'indexation d'images
# ============================

@app.post("/index_images/", response_model=IndexingResult)
async def index_images(file: UploadFile = File(...)):
    """
    Cette route permet d'indexer une image en extrayant ses caractéristiques 
    et en l'ajoutant à l'index d'images (ElasticSearch).
    """
    try:
        # Valider et sauvegarder l'image
        image_path = f"data/images/{file.filename}"
        with open(image_path, "wb") as f:
            f.write(await file.read())
        
        # Vérification de l'image
        if not validate_image_file(image_path):
            raise HTTPException(status_code=400, detail="Fichier image invalide.")
        
        # Extraction des caractéristiques de l'image
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(image_path)
        
        # Indexation de l'image
        indexer = ImageIndexerManager()
        image_id = indexer.index_image(image_path, features)
        
        return IndexingResult(status="success", message=f"Image {image_id} indexée avec succès.")
    
    except Exception as e:
        handle_error(f"Erreur lors de l'indexation de l'image: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

# ============================
# Point de terminaison pour la recherche d'images similaires
# ============================

@app.post("/search_images/", response_model=List[ImageMetadata])
async def search_images(query: SearchQuery):
    """
    Cette route permet de rechercher des images similaires à une image de requête.
    """
    try:
        # Validation de l'image de requête
        if not validate_image_file(query.query_image):
            raise HTTPException(status_code=400, detail="Fichier image de requête invalide.")
        
        # Extraction des caractéristiques de l'image de requête
        feature_extractor = FeatureExtractor()
        query_features = feature_extractor.extract_features(query.query_image)
        
        # Recherche des images similaires
        indexer = ImageIndexerManager()
        results = indexer.search_similar_images(query_features, top_k=query.top_k)
        
        # Récupérer les métadonnées des images similaires
        similar_images = []
        for result in results:
            similar_images.append(ImageMetadata(image_id=result["image_id"], features=result["features"]))
        
        return similar_images
    
    except Exception as e:
        handle_error(f"Erreur lors de la recherche d'images similaires: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

# ============================
# Point de terminaison pour la gestion des erreurs
# ============================

@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    """
    Fonction pour gérer les exceptions globales et retourner une réponse
    d'erreur formatée pour l'API.
    """
    logger.error(f"Une erreur est survenue: {str(exc)}")
    return {"error": str(exc)}

# ============================
# Lancer l'application
# ============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
