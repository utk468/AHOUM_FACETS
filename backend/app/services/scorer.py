import os
import re
import csv
import numpy as np
from typing import Dict, Any, List
from backend.app.database import get_db_client
from backend.app.services.embedding import EmbeddingService

class FacetScorerService:
    _cached_facets = None

    @classmethod
    def get_facets_from_db_or_fallback(cls) -> List[Dict[str, Any]]:
        # Singleton cache to keep API requests extremely fast
        if cls._cached_facets is not None:
            return cls._cached_facets
            
        db = get_db_client()
        if db is not None:
            try:
                facets_coll = db["facets"]
                facets = list(facets_coll.find({}, {"_id": 0, "raw_name": 1, "name": 1, "category": 1, "description": 1, "weights": 1, "embedding": 1}))
                if len(facets) > 0:
                    cls._cached_facets = facets
                    print(f"[+] Loaded {len(facets)} facets from MongoDB database.")
                    return facets
            except Exception as e:
                print(f"[x] Failed to load facets from DB: {e}. Falling back to CSV processing...")
                
        # CSV direct loading fallback
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        csv_path = os.path.join(base_dir, "Facets Assignment.csv")
        
        if not os.path.exists(csv_path):
            print(f"[x] Critical: CSV file not found at {csv_path}. Returning empty list.")
            return []
            
        print(f"[*] Falling back to direct load from CSV: {csv_path}")
        try:
            raw_names = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row and row[0]:
                        raw_names.append(str(row[0]))
            
            # Clean and deduplicate
            seen = set()
            unique_facets = []
            for r in raw_names:
                clean = re.sub(r'^\d+\.\s*', '', r).strip().rstrip(':')
                if clean and clean not in seen:
                    seen.add(clean)
                    unique_facets.append((r, clean))
            
            # Formulate mock/fallback facets with weights
            from backend.app.services.extractor import CORE_FEATURE_KEYS
            facets = []
            import hashlib
            for r, c in unique_facets:
                # Basic mock weights
                hash_hex = hashlib.md5(c.encode('utf-8')).hexdigest()
                seed = int(hash_hex, 16) & 0xffffffff
                rng = np.random.default_rng(seed)
                primary_idx = rng.integers(0, len(CORE_FEATURE_KEYS))
                weights = [0.0] * len(CORE_FEATURE_KEYS)
                weights[primary_idx] = 0.8
                for idx in range(len(weights)):
                    if idx != primary_idx:
                        weights[idx] = 0.2 / (len(CORE_FEATURE_KEYS) - 1)
                        
                facets.append({
                    "raw_name": r,
                    "name": c,
                    "category": "Behavioral",
                    "description": f"Measures {c.lower()} in conversation.",
                    "weights": weights,
                    "embedding": [0.0] * 384
                })
                
            cls._cached_facets = facets
            print(f"[+] Loaded {len(facets)} facets from CSV fallback successfully!")
            return facets
        except Exception as csv_err:
            print(f"[x] Failed to load CSV fallback: {csv_err}")
            return []

    @classmethod
    def score_facets(cls, turn_text: str, core_features: Dict[str, float], turn_embedding: List[float]) -> Dict[str, Any]:
        facets = cls.get_facets_from_db_or_fallback()
        
        # Turn features into sorted numpy array
        from backend.app.services.extractor import CORE_FEATURE_KEYS
        feature_vector = np.array([core_features.get(key, 0.5) for key in CORE_FEATURE_KEYS])
        
        scores_result = {}
        
        # Convert all facet weights to matrix of size N x 30
        weights_matrix = np.array([f["weights"] for f in facets])
        
        # Matrix multiplication
        raw_scores = np.dot(weights_matrix, feature_vector)
        
        # Turn-facet similarity blending
        t_vec = np.array(turn_embedding)
        t_norm = t_vec / (np.linalg.norm(t_vec) + 1e-9)
        
        facet_embeddings_list = [f.get("embedding", [0.0] * 384) for f in facets]
        has_embeddings = any(np.sum(e) != 0.0 for e in facet_embeddings_list)
        
        similarities = np.zeros(len(facets))
        if has_embeddings:
            embeddings_matrix = np.array(facet_embeddings_list)
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-9
            normalized_embeddings = embeddings_matrix / norms
            similarities = np.dot(normalized_embeddings, t_norm)
            
        for i, f in enumerate(facets):
            raw_score = raw_scores[i]
            sim = similarities[i] if has_embeddings else 0.0
            
            semantic_blend = max(0.0, sim)
            blended_score = 0.8 * raw_score + 0.2 * semantic_blend
            
            mapped_score = 1.0 + 4.0 * blended_score
            
            if core_features.get("_is_mock", False):
                import hashlib
                hash_hex = hashlib.md5((turn_text + f["name"]).encode('utf-8')).hexdigest()
                facet_rng = np.random.default_rng(int(hash_hex, 16) & 0xffffffff)
                mapped_score += facet_rng.uniform(-1.8, 1.8)
                
            final_score_int = int(np.clip(round(mapped_score), 1, 5))
            
            confidence = 0.80 + 0.15 * max(0, sim) + 0.05 * (raw_score - 0.5)
            confidence = float(np.clip(confidence, 0.60, 0.99))
            
            scores_result[f["name"]] = {
                "score": final_score_int,
                "confidence": round(confidence, 2)
            }
            
        return scores_result
