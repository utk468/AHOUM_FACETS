import numpy as np
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("[!] sentence-transformers not installed. Embedding service will run in MOCK mode.")

class EmbeddingService:
    _model = None

    @classmethod
    def get_model(cls):
        if not EMBEDDING_AVAILABLE:
            return None
        if cls._model is None:
            try:
                print("[*] Loading local sentence embedding model 'all-MiniLM-L6-v2'...")
                cls._model = SentenceTransformer('all-MiniLM-L6-v2')
                print("[+] Embedding model loaded successfully!")
            except Exception as e:
                print(f"[x] Failed to load embedding model: {e}")
                cls._model = None
        return cls._model

    @classmethod
    def get_embedding(cls, text: str) -> List[float]:
        model = cls.get_model()
        if model is not None:
            try:
                embedding = model.encode(text)
                return embedding.tolist()
            except Exception as e:
                print(f"[x] Error encoding text: {e}")
        
        # MOCK Fallback: return a pseudo-random deterministic vector of size 384
        # based on character hash of text
        import hashlib
        hash_hex = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(hash_hex, 16) & 0xffffffff
        rng = np.random.default_rng(seed)
        mock_vec = rng.normal(0.0, 1.0, 384)
        mock_vec = mock_vec / np.linalg.norm(mock_vec)
        return mock_vec.tolist()
