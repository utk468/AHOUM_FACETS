from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class ConversationTurnInput(BaseModel):
    user: str = Field(..., description="The user's query or message")
    assistant: str = Field(..., description="The assistant's response to evaluate")
    conversation_id: Optional[str] = Field(None, description="Identifier for the conversation thread")
    turn_id: Optional[int] = Field(1, description="Index of this turn in the conversation")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="Previous turns in format [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]")

class TurnMetadata(BaseModel):
    turn_length: int = Field(..., description="Number of words in assistant response")
    sentiment_score: str = Field(..., description="Emotion detection status")
    toxicity_score: float = Field(..., description="Safety / toxicity probability")
    readability_score: float = Field(..., description="Language quality / Flesch Reading Ease score")
    response_time_estimate: float = Field(..., description="Estimated processing complexity in seconds")
    topic: str = Field(..., description="Conversation domain category")
    intent: str = Field(..., description="User's primary intent")
    confidence: float = Field(..., description="Model evaluation confidence")
    conversation_depth: int = Field(..., description="Index of this turn in dialogue")
    context_window_size: int = Field(..., description="Context complexity in character count")

class FacetScoreDetail(BaseModel):
    score: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0.0, le=1.0)

class ConversationTurnOut(BaseModel):
    id: Optional[str] = None
    conversation_id: str
    turn_id: int
    user: str
    assistant: str
    metadata: TurnMetadata
    facet_scores: Dict[str, FacetScoreDetail]
    created_at: datetime
