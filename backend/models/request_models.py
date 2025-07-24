from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Literal
from datetime import datetime

class RetrieveNodesRequest(BaseModel):
    query: str
    index: str
    top_k: int = 5
    mode: Literal['vector', 'keyword', 'hybrid'] = 'hybrid'
    rerank: bool = True
    use_graph: bool = False

class RetrievedNode(BaseModel):
    node_id: str
    node_score: float
    node_text: str
    node_metadata: Dict[str, Any]

class RetrieveNodesResponse(BaseModel):
    nodes: List[RetrievedNode]
    context_tokens: int
    query: str
    
class QueryIndexRequest(BaseModel):
    query: str
    index: str
    top_k: int = 5
    mode: Literal['vector', 'keyword', 'hybrid'] = 'hybrid'
    rerank: bool = True
    use_graph: bool = False
    include_sources: bool = True

class QueryIndexResponse(BaseModel):
    response: str
    sources: Optional[List[RetrievedNode]] = None
    query: str
    context_tokens: int
    metadata: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int

class CreateIndexRequest(BaseModel):
    index_name: str
    description: str
    source_url: str
    source_type: Literal['github', 'website', 'other'] = 'documentation'
    requester_name: Optional[str] = None
    requester_email: Optional[str] = None
    additional_notes: Optional[str] = None

class CreateIndexResponse(BaseModel):
    request_id: str
    message: str
    status: str
    submitted_at: datetime