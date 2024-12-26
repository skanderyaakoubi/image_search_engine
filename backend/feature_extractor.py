import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv
from scipy.stats import entropy
from typing import Optional, Union, Tuple
import tensorflow as tf

class EnhancedFeatureExtractor:
    def __init__(self):
        # Initialiser VGG16
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        
        # Paramètres pour LBP (Local Binary Pattern)
        self.n_points = 8
        self.radius = 1
        
        # Paramètres pour l'histogramme de couleur
        self.color_bins = 32
        
        # Initialiser SIFT
        self.sift = cv2.SIFT_create()
        
        # Dimensions pour chaque type de feature
        self.vgg_dims = 3072  # Réduit pour faire de la place aux autres features
        self.color_dims = 512
        self.texture_dims = 256
        self.sift_dims = 256
        
        # Vérifier que la somme fait 4096
        assert self.vgg_dims + self.color_dims + self.texture_dims + self.sift_dims == 4096

    def extract_color_features(self, img: np.ndarray) -> np.ndarray:
        """Extraire les caractéristiques de couleur"""
        # Convertir en HSV
        hsv = rgb2hsv(img)
        
        # Calculer les histogrammes pour chaque canal HSV
        h_hist = np.histogram(hsv[..., 0], bins=self.color_bins, range=(0, 1))[0]
        s_hist = np.histogram(hsv[..., 1], bins=self.color_bins, range=(0, 1))[0]
        v_hist = np.histogram(hsv[..., 2], bins=self.color_bins, range=(0, 1))[0]
        
        # Calculer des statistiques de couleur
        mean_color = np.mean(hsv, axis=(0, 1))
        std_color = np.std(hsv, axis=(0, 1))
        skew_color = np.mean((hsv - np.mean(hsv, axis=(0, 1))) ** 3, axis=(0, 1))
        
        # Combiner toutes les caractéristiques de couleur
        color_features = np.concatenate([
            h_hist / np.sum(h_hist),
            s_hist / np.sum(s_hist),
            v_hist / np.sum(v_hist),
            mean_color,
            std_color,
            skew_color
        ])
        
        # Redimensionner à color_dims
        return self._resize_features(color_features, self.color_dims)

    def extract_texture_features(self, img: np.ndarray) -> np.ndarray:
        """Extraire les caractéristiques de texture"""
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Calculer LBP
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method='uniform')
        lbp_hist = np.histogram(lbp, bins=self.n_points + 2, range=(0, self.n_points + 2))[0]
        
        # Calculer la matrice de co-occurrence
        glcm = self._compute_glcm(gray)
        
        # Calculer des statistiques de texture
        contrast = np.sum((np.arange(256) - np.arange(256)[:, np.newaxis]) ** 2 * glcm)
        correlation = np.sum(glcm * np.outer(np.arange(256), np.arange(256)))
        energy = np.sum(glcm ** 2)
        homogeneity = np.sum(glcm / (1 + np.abs(np.arange(256) - np.arange(256)[:, np.newaxis])))
        
        # Combiner les caractéristiques de texture
        texture_features = np.concatenate([
            lbp_hist / np.sum(lbp_hist),
            [contrast, correlation, energy, homogeneity]
        ])
        
        # Redimensionner à texture_dims
        return self._resize_features(texture_features, self.texture_dims)

    def extract_sift_features(self, img: np.ndarray) -> np.ndarray:
        """Extraire les caractéristiques SIFT"""
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Détecter et calculer les descripteurs SIFT
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            return np.zeros(self.sift_dims)
        
        # Calculer un "bag of visual words" simplifié
        if len(descriptors) > self.sift_dims:
            # Sous-échantillonner si trop de descripteurs
            indices = np.linspace(0, len(descriptors)-1, self.sift_dims, dtype=int)
            descriptors = descriptors[indices]
        
        # Créer un vecteur de taille fixe
        sift_features = np.zeros(self.sift_dims)
        sift_features[:len(descriptors)] = np.mean(descriptors, axis=1)
        
        return self._resize_features(sift_features, self.sift_dims)

    def _compute_glcm(self, gray: np.ndarray, distance: int = 1, angles: list = [0]) -> np.ndarray:
        """Calculer la matrice de co-occurrence des niveaux de gris"""
        glcm = np.zeros((256, 256))
        for angle in angles:
            dx = int(distance * np.cos(angle))
            dy = int(distance * np.sin(angle))
            for i in range(gray.shape[0] - dx):
                for j in range(gray.shape[1] - dy):
                    i_value = gray[i, j]
                    j_value = gray[i+dx, j+dy]
                    glcm[i_value, j_value] += 1
        return glcm / np.sum(glcm)

    def _resize_features(self, features: np.ndarray, target_size: int) -> np.ndarray:
        """Redimensionner un vecteur de caractéristiques à une taille cible"""
        if len(features) == target_size:
            return features
        elif len(features) > target_size:
            # Sous-échantillonner
            return features[:target_size]
        else:
            # Padding avec des zéros
            padded = np.zeros(target_size)
            padded[:len(features)] = features
            return padded

    def extract(self, img) -> np.ndarray:
        """
        Extraire toutes les caractéristiques et les combiner
        
        Args:
            img: PIL Image ou chemin d'image
            
        Returns:
            np.ndarray: Vecteur de caractéristiques de dimension 4096
        """
        # Prétraiter l'image pour VGG16
        if isinstance(img, str):
            img = image.load_img(img, target_size=(224, 224))
        img_vgg = img.resize((224, 224))
        img_vgg = img_vgg.convert('RGB')
        x = image.img_to_array(img_vgg)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Extraire les features VGG16
        vgg_features = self.model.predict(x)[0]
        vgg_features = vgg_features[:self.vgg_dims]  # Prendre seulement vgg_dims features
        
        # Convertir l'image en numpy array pour les autres extracteurs
        img_array = np.array(img_vgg)
        
        # Extraire les autres caractéristiques
        color_features = self.extract_color_features(img_array)
        texture_features = self.extract_texture_features(img_array)
        sift_features = self.extract_sift_features(img_array)
        
        # Normaliser chaque groupe de caractéristiques
        vgg_features = vgg_features / (np.linalg.norm(vgg_features) + 1e-7)
        color_features = color_features / (np.linalg.norm(color_features) + 1e-7)
        texture_features = texture_features / (np.linalg.norm(texture_features) + 1e-7)
        sift_features = sift_features / (np.linalg.norm(sift_features) + 1e-7)
        
        # Combiner toutes les caractéristiques
        combined_features = np.concatenate([
            vgg_features,      # 3072 dimensions
            color_features,    # 512 dimensions
            texture_features,  # 256 dimensions
            sift_features     # 256 dimensions
        ])
        
        # Normalisation finale
        combined_features = combined_features / np.linalg.norm(combined_features)
        
        return combined_features