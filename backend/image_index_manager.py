from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
import hashlib
from PIL import Image
import numpy as np
from elasticsearch import Elasticsearch, helpers
from feature_extractor import EnhancedFeatureExtractor
from concurrent.futures import ThreadPoolExecutor
import mimetypes
from elasticsearch.helpers import scan
import imagehash
from tqdm import tqdm
import matplotlib.pyplot as plt

class ImageIndexManager:
    def __init__(self, es_host: str = 'localhost', es_port: int = 9200, index_name: str = 'images_vgg166'):
        self.es = Elasticsearch([{'host': es_host, 'port': es_port}])
        self.fe = EnhancedFeatureExtractor()  # Classe pour l'extraction des caractéristiques
        self.index_name = index_name
        self.setup_logging()
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('cbir.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_index(self) -> None:
        settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "index.mapping.total_fields.limit": 2000,
                "index.max_result_window": 10000,
                "analysis": {
                    "analyzer": {
                        "tag_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "image_vector": {
                        "type": "dense_vector",
                        "dims": 4096,  # Adapté à l'EnhancedFeatureExtractor
                        "similarity": "cosine"
                    },
                    "image_path": {"type": "keyword"},
                    "image_hash": {"type": "keyword"},
                    "perceptual_hash": {"type": "keyword"},
                    "metadata": {
                        "properties": {
                            "title": {"type": "text", "analyzer": "tag_analyzer"},
                            "tags": {"type": "keyword", "normalizer": "lowercase"},
                            "date_indexed": {"type": "date"},
                            "file_size": {"type": "long"},
                            "mime_type": {"type": "keyword"},
                            "dimensions": {
                                "properties": {
                                    "width": {"type": "integer"},
                                    "height": {"type": "integer"},
                                    "aspect_ratio": {"type": "float"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name, body=settings)
                self.logger.info(f"Index '{self.index_name}' created successfully")
            else:
                self.logger.warning(f"Index '{self.index_name}' already exists")
        except Exception as e:
            self.logger.error(f"Failed to create index: {e}")
            raise

    def calculate_image_hashes(self, img: Image) -> Tuple[str, str]:
        """Calculer les hashes cryptographique et perceptuel de l'image"""
        img_array = np.array(img)
        crypto_hash = hashlib.md5(img_array.tobytes()).hexdigest()
        phash = str(imagehash.average_hash(img))
        return crypto_hash, phash

    def process_image(self, image_path: Path) -> Optional[Dict]:
        """Traiter une seule image et préparer son document pour l'indexation"""
        try:
            if image_path.suffix.lower() not in self.supported_formats:
                self.logger.warning(f"Unsupported format: {image_path}")
                return None

            img = Image.open(image_path)
            
            # Extraire les caractéristiques avec EnhancedFeatureExtractor
            feature = self.fe.extract(img)
            
            # Calculer les hashes et les métadonnées
            crypto_hash, phash = self.calculate_image_hashes(img)
            file_size = image_path.stat().st_size
            mime_type = mimetypes.guess_type(image_path)[0]
            
            doc = {
                "image_vector": feature.tolist(),
                "image_path": str(image_path),
                "image_hash": crypto_hash,
                "perceptual_hash": phash,
                "metadata": {
                    "title": image_path.stem,
                    "tags": [],
                    "date_indexed": datetime.now().isoformat(),
                    "file_size": file_size,
                    "mime_type": mime_type,
                    "dimensions": {
                        "width": img.width,
                        "height": img.height,
                        "aspect_ratio": img.width / img.height
                    }
                }
            }
            return doc
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return None

    def bulk_index_images(self, image_dir: str, batch_size: int = 249, max_workers: int = 2) -> None:
        image_dir = Path(image_dir)
        image_paths = list(image_dir.rglob("*.*"))
        total_images = len(image_paths)
    
        self.logger.info(f"Found {total_images} files to process")
    
        # Traiter les images par lots pour éviter la surcharge mémoire
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            processed_docs = []
            failed_paths = []
        
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for img_path in batch_paths:
                    if img_path.suffix.lower() in self.supported_formats:
                        futures.append(executor.submit(self.process_image, img_path))
            
                with tqdm(total=len(futures), desc=f"Processing batch {i//batch_size + 1}") as pbar:
                    for future in futures:
                        try:
                            doc = future.result(timeout=30)  # Timeout après 30 secondes
                            if doc:
                                processed_docs.append({
                                "_index": self.index_name,
                                "_source": doc
                                })
                            else:
                                failed_paths.append(str(img_path))
                        except Exception as e:
                            self.logger.error(f"Failed to process image: {e}")
                            failed_paths.append(str(img_path))
                        finally:
                            pbar.update(1)

            if processed_docs:
                try:
                    success, failed = helpers.bulk(
                        self.es,
                        processed_docs,
                        chunk_size=min(batch_size, 25),
                        raise_on_error=False,
                        request_timeout=30  # Timeout pour les requêtes Elasticsearch
                    )
                    self.logger.info(f"Batch {i//batch_size + 1}: Indexed {success} images successfully. Failed: {len(failed) if failed else 0}")
                except Exception as e:
                    self.logger.error(f"Bulk indexing failed for batch {i//batch_size + 1}: {e}")
                    failed_paths.extend([doc["_source"]["image_path"] for doc in processed_docs])

            # Sauvegarder les documents non indexés
            if failed_paths:
                with open('failed_images.log', 'a') as f:
                    for path in failed_paths:
                        f.write(f"{path}\n")

        self.logger.info("Bulk indexing completed")

    def search_similar_images(
        self,
        query_img: Image,
        size: int = 5,
        min_score: float = 0.5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Rechercher des images similaires"""
        query_vector = self.fe.extract(query_img)
        
        must_conditions = []
        if filters:
            for key, value in filters.items():
                must_conditions.append({"term": {key: value}})

        search_query = {
            "size": size,
            "min_score": min_score,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "must": must_conditions or {"match_all": {}}
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'image_vector') + 1.0",
                        "params": {
                            "query_vector": query_vector.tolist()
                        }
                    }
                }
            },
            "_source": ["image_path", "metadata", "image_hash", "perceptual_hash"]
        }
        
        try:
            response = self.es.search(index=self.index_name, body=search_query)
            results = [{
                "score": hit["_score"],
                "image_path": hit["_source"]["image_path"],
                "metadata": hit["_source"]["metadata"],
                "image_hash": hit["_source"]["image_hash"],
                "perceptual_hash": hit["_source"]["perceptual_hash"]
            } for hit in response["hits"]["hits"]]
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            results = []

        # Afficher l'image requête et les images similaires
        plt.figure(figsize=(15, 8))
        plt.subplot(1, size + 1, 1)
        plt.imshow(query_img)
        plt.title('Image Requête')
        plt.axis('off')
    
        for idx, result in enumerate(results, 1):
            try:
                similar_img = Image.open(result["image_path"])
                plt.subplot(1, size + 1, idx + 1)
                plt.imshow(similar_img)
                plt.title(f'Score: {result["score"]:.2f}')
                plt.axis('off')
            except Exception as e:
                self.logger.error(f"Error displaying {result['image_path']}: {e}")
                continue
    
        plt.tight_layout()
        plt.show()
    
        return results

    def delete_duplicate_images(self, similarity_threshold: float = 0.95) -> None:
        """Supprimer les images en double"""
        try:
            all_docs = [doc for doc in scan(self.es, index=self.index_name, _source=True)]
            duplicates_to_delete = set()
            
            hash_groups = {}
            for doc in all_docs:
                phash = doc["_source"]["perceptual_hash"]
                hash_groups.setdefault(phash, []).append(doc)
            
            for docs in hash_groups.values():
                if len(docs) > 1:
                    base_doc = docs[0]
                    base_vector = np.array(base_doc["_source"]["image_vector"])
                    
                    for doc in docs[1:]:
                        vector = np.array(doc["_source"]["image_vector"])
                        similarity = np.dot(base_vector, vector)
                        
                        if similarity >= similarity_threshold:
                            duplicates_to_delete.add(doc["_id"])
            
            if duplicates_to_delete:
                for doc_id in duplicates_to_delete:
                    self.es.delete(index=self.index_name, id=doc_id)
                self.logger.info(f"Deleted {len(duplicates_to_delete)} duplicate images")
            else:
                self.logger.info("No duplicates found")
                
        except Exception as e:
            self.logger.error(f"Error during duplicate deletion: {e}")
