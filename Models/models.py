from typing import Dict, List
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    text: str
    chunk_size : int

class overlappingChunk(BaseModel):
    text : str
    chunk_size : int
    overlapping_size : int

class messageSchema(BaseModel):
    user_question: str
    context_chunks: List[Dict]

class Citation(BaseModel):
    doc_id: str
    chunk_id: int

class AnswerSchema(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: List[Citation] = []
    evidence: List[str] = [] 
