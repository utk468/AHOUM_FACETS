import uuid
from datetime import datetime
from typing import Dict, Any
from backend.app.models.conversation import ConversationTurnInput, TurnMetadata, FacetScoreDetail
from backend.app.services.embedding import EmbeddingService
from backend.app.services.extractor import FeatureExtractorService
from backend.app.services.scorer import FacetScorerService
from backend.app.database import get_db_client

class EvaluationPipeline:

    @classmethod
    def run_evaluation(cls, payload: ConversationTurnInput) -> Dict[str, Any]:
        conversation_id = payload.conversation_id or str(uuid.uuid4())
        turn_id = payload.turn_id or 1
        
        # Text Embedding of the response turn
        text_to_embed = f"User: {payload.user}\nAssistant: {payload.assistant}"
        turn_embedding = EmbeddingService.get_embedding(text_to_embed)
        
        # Feature & Metadata Extraction (using Groq Llama-3/Qwen)
        extracted = FeatureExtractorService.extract_features(
            user_text=payload.user,
            assistant_text=payload.assistant,
            history=payload.history
        )
        
        core_features = extracted["core_features"]
        metadata_dict = extracted["metadata"]
        
        # Scalable Facet Scoring via Matrix dot product
        facet_scores = FacetScorerService.score_facets(
            turn_text=payload.assistant,
            core_features=core_features,
            turn_embedding=turn_embedding
        )
        
        # Construct output
        metadata = TurnMetadata(
            turn_length=metadata_dict["turn_length"],
            sentiment_score=metadata_dict["sentiment_score"],
            toxicity_score=metadata_dict["toxicity_score"],
            readability_score=metadata_dict["readability_score"],
            response_time_estimate=metadata_dict["response_time_estimate"],
            topic=metadata_dict["topic"],
            intent=metadata_dict["intent"],
            confidence=metadata_dict["confidence"],
            conversation_depth=turn_id,
            context_window_size=metadata_dict["context_window_size"]
        )
        
        evaluation_record = {
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "user": payload.user,
            "assistant": payload.assistant,
            "metadata": metadata.model_dump(),
            "facet_scores": {k: {"score": v["score"], "confidence": v["confidence"]} for k, v in facet_scores.items()},
            "created_at": datetime.utcnow()
        }
        
        # Save to MongoDB
        db = get_db_client()
        if db is not None:
            try:
                conversations_coll = db["conversations"]
                conversations_coll.update_one(
                    {"conversation_id": conversation_id, "turn_id": turn_id},
                    {"$set": evaluation_record},
                    upsert=True
                )
                print(f"[+] Saved turn evaluation to MongoDB for conversation {conversation_id}, turn {turn_id}")
            except Exception as e:
                print(f"[x] Failed to save evaluation to MongoDB: {e}")
                
        response_record = evaluation_record.copy()
        if "_id" in response_record:
            response_record["id"] = str(response_record["_id"])
            del response_record["_id"]
        else:
            response_record["id"] = f"{conversation_id}_{turn_id}"
            
        return response_record
