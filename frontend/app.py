# frontend/app.py

import streamlit as st
from PIL import Image
import requests
import json
from .utils import validate_image_file, upload_image, show_error

# ============================
# Titre de l'application
# ============================
st.title("Content-Based Image Retrieval (CBIR)")
st.write("Bienvenue dans l'application CBIR. Vous pouvez indexer des images et rechercher des images similaires.")

# ============================
# Section pour l'indexation d'images
# ============================
st.header("Indexation d'images")
uploaded_image = st.file_uploader("Téléchargez une image pour l'indexer", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Vérifier si le fichier est une image valide
    if validate_image_file(uploaded_image):
        st.image(uploaded_image, caption="Image téléchargée", use_column_width=True)

        # Bouton pour indexer l'image
        if st.button("Indexer l'image"):
            result = upload_image(uploaded_image)
            if isinstance(result, dict):
                st.success(f"Image indexée avec succès : {result['message']}")
            else:
                st.error(result)
    else:
        st.error("Le fichier téléchargé n'est pas une image valide.")

# ============================
# Section pour la recherche d'images similaires
# ============================
st.header("Recherche d'images similaires")
query_image = st.file_uploader("Téléchargez une image de requête", type=["jpg", "png", "jpeg"])

if query_image is not None:
    # Vérifier si le fichier est une image valide
    if validate_image_file(query_image):
        st.image(query_image, caption="Image de requête", use_column_width=True)

        # Bouton pour lancer la recherche
        if st.button("Rechercher des images similaires"):
            try:
                # Charger l'image et envoyer une requête à l'API backend pour la recherche
                response = requests.post(
                    "http://localhost:8000/search_images/",
                    json={"query_image": query_image.name, "top_k": 5}
                )
                if response.status_code == 200:
                    results = response.json()
                    if results:
                        st.write("Résultats de la recherche :")
                        for result in results:
                            st.image(result['image_id'], caption=f"Image ID: {result['image_id']}", use_column_width=True)
                    else:
                        st.write("Aucune image similaire trouvée.")
                else:
                    st.error("Erreur dans la recherche.")
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion au backend : {e}")
    else:
        st.error("Le fichier téléchargé n'est pas une image valide.")
