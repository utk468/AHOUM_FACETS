import numpy as np
from typing import List
from huggingface_hub import InferenceClient
from backend.app.config import settings

class EmbeddingService:
    _client = None

    @classmethod
    def get_client(cls):
        if not settings.HF_TOKEN or settings.HF_TOKEN == "YOUR_HF_TOKEN":
            return None
            
        if cls._client is None:
            try:
                print("[*] Initializing HuggingFace InferenceClient...")
                cls._client = InferenceClient(api_key=settings.HF_TOKEN)
            except Exception as e:
                print(f"[x] Failed to initialize HF InferenceClient: {e}")
                cls._client = None
        return cls._client

    @classmethod
    def get_embedding(cls, text: str) -> List[float]:
        client = cls.get_client()
        if client is not None:
            try:
                # The model returns a list of floats (embedding) for a single input
                embedding = client.feature_extraction(
                    text,
                    model="jinaai/jina-embeddings-v5-text-nano"
                )
                
                # Check if it's nested (batch response format)
                if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                    return embedding[0]
                return embedding
            except Exception as e:
                print(f"[x] Error encoding text via HF API: {e}")
        
        # MOCK Fallback: return a pseudo-random deterministic vector of size 384
        # based on character hash of text
        import hashlib
        hash_hex = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(hash_hex, 16) & 0xffffffff
        rng = np.random.default_rng(seed)
        mock_vec = rng.normal(0.0, 1.0, 384)
        mock_vec = mock_vec / np.linalg.norm(mock_vec)
        return mock_vec.tolist()
