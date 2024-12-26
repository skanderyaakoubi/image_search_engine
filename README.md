<<<<<<< HEAD
# ImageSearchEngine
=======
# Content-Based Image Retrieval (CBIR)

## Description

Le projet **Content-Based Image Retrieval (CBIR)** permet de rechercher des images similaires à une image de requête en utilisant des caractéristiques extraites des images indexées dans une base de données Elasticsearch. Le système comprend un backend FastAPI pour gérer les requêtes d'indexation et de recherche d'images, ainsi qu'un frontend Streamlit pour l'interaction avec l'utilisateur.

## Fonctionnalités

- **Indexation des images** : Téléchargez une image, et le système extrait ses caractéristiques, puis les indexe dans Elasticsearch pour permettre la recherche rapide.
- **Recherche d'images similaires** : Soumettez une image de requête et récupérez les images les plus similaires en fonction des caractéristiques extraites.
- **Frontend interactif avec Streamlit** : Interface utilisateur simple permettant de télécharger des images, de lancer la recherche et de visualiser les résultats.

## Architecture

Le projet est divisé en deux parties principales :

### Backend (FastAPI)
- **API pour indexer et rechercher des images**.
- Utilisation d'**Elasticsearch** pour indexer les images et leurs caractéristiques.
- **Extraction des caractéristiques** des images via des méthodes de traitement d'image.

### Frontend (Streamlit)
- Interface utilisateur pour télécharger une image, afficher les résultats de recherche, et interagir avec l'API FastAPI.

## Prérequis

Avant de démarrer, assurez-vous d'avoir les éléments suivants installés :
- **Python 3.x**
- **Elasticsearch** (localement ou via un service distant comme Elastic Cloud)
- **pip** pour installer les dépendances

## Installation

### Étape 1 : Clonez le dépôt

Clonez le dépôt Git sur votre machine locale :

```bash
git clone https://github.com/yourusername/cbir_project.git
cd cbir_project
>>>>>>> ff41d0b (Premier commit)
