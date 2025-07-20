from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import sys
import os
import uuid
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import your existing functions
try:
    from core.retriever import retrieve_nodes, query_index, get_nodes_tokens
except ImportError as e:
    print(f"Error importing from core.retriever: {e}")
    # Fallback for development - you can remove this in production
    # def retrieve_nodes(*args, **kwargs):
    #     raise HTTPException(status_code=500, detail="retrieve_nodes function not available")
    
    # def query_index(*args, **kwargs):
    #     raise HTTPException(status_code=500, detail="query_index function not available")

from backend.models.request_models import (
    RetrieveNodesRequest, 
    RetrieveNodesResponse,
    QueryIndexRequest,
    QueryIndexResponse,
    ErrorResponse,
    RetrievedNode,
    CreateIndexRequest,
    CreateIndexResponse
)

router = APIRouter()

@router.post("/retrieve-nodes", response_model=RetrieveNodesResponse)
async def api_retrieve_nodes(request: RetrieveNodesRequest):
    """
    Retrieve nodes based on query using your existing retrieve_nodes function
    """
    try:
        nodes = await retrieve_nodes(
            query=request.query,
            index=request.index,
            top_k=request.top_k,
            mode=request.mode,
            rerank=request.rerank,
            use_graph=request.use_graph
        )

        retrieved_nodes = [RetrievedNode(node_id=node.node.node_id, 
                                         node_score=node.score, 
                                         node_text=node.node.text, 
                                         node_metadata=node.node.metadata)
                                           for node in nodes]

        context_tokens = get_nodes_tokens(nodes)
        
        return RetrieveNodesResponse(
            nodes=retrieved_nodes,
            context_tokens=context_tokens,
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving nodes: {str(e)}")

@router.post("/query-index", response_model=QueryIndexResponse)
async def api_query_index(request: QueryIndexRequest):
    """
    Query the index using your existing query_index function
    """
    try:
        # Call your existing function
        # Adjust parameters based on your actual function signature
        result = await query_index(
            query=request.query,
            index=request.index,
            top_k=request.top_k,
            mode=request.mode,
            rerank=request.rerank,
            use_graph=request.use_graph,
        )
        
        retrieved_nodes = [RetrievedNode(node_id=node.node.node_id, 
                                                node_score=node.score, 
                                                node_text=node.node.text, 
                                                node_metadata=node.node.metadata)
                                                for node in result.source_nodes]        
        context_tokens = get_nodes_tokens(result.source_nodes)
        
        return QueryIndexResponse(
            response=result.response,
            sources=retrieved_nodes if request.include_sources else None,
            query=request.query,
            metadata=result.metadata,
            context_tokens=context_tokens
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying index: {str(e)}")

@router.post("/request-index", response_model=CreateIndexResponse)
async def request_index_creation(request: CreateIndexRequest, http_request: Request):
    """
    Submit a request for creating a new index
    """
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Get client IP address
        client_ip = http_request.client.host if http_request.client else "unknown"
        # Check for forwarded IP in case of proxy
        forwarded_for = http_request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        # Create request data
        request_data = {
            "request_id": request_id,
            "index_name": request.index_name,
            "description": request.description,
            "source_url": request.source_url,
            "source_type": request.source_type,
            "requester_name": request.requester_name,
            "requester_email": request.requester_email,
            "additional_notes": request.additional_notes,
            "client_ip": client_ip,
            "submitted_at": timestamp.isoformat(),
            "status": "pending"
        }
        
        # Determine the processed data path
        processed_data_path = os.environ.get('PROCESSED_DATA_PATH', 'processed_data')
        requests_file = Path(processed_data_path) / "index_requests.csv"
        
        # Ensure the directory exists
        requests_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Define CSV headers
        csv_headers = [
            "request_id", "index_name", "description", "source_url", "source_type",
            "requester_name", "requester_email", "additional_notes", "client_ip",
            "submitted_at", "status"
        ]
        
        # Check if file exists to determine if we need to write headers
        file_exists = requests_file.exists()
        
        # Append the request to the CSV file
        with open(requests_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            
            # Write headers if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write the request data
            writer.writerow(request_data)
        
        return CreateIndexResponse(
            request_id=request_id,
            message=f"Index creation request for '{request.index_name}' has been submitted successfully",
            status="submitted",
            submitted_at=timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting index request: {str(e)}")

@router.get("/indexes")
async def get_available_indexes():
    """
    Get list of available indexes (hard-coded for now)
    """
    # Hard-coded indexes based on your processed_data directory structure
    available_indexes = [
        {
            "name": "google_genai-api",
            "description": "Google Genai Gemini api documentation",
            "source": "https://ai.google.dev/api"
        },
        {
            "name": "google_genai-docs", 
            "description": "Google Genai Gemini documentation",
            "source": "https://ai.google.dev/gemini-api/docs"
        },
    ]
    
    return {
        "indexes": available_indexes,
        "count": len(available_indexes)
    }

@router.get("/status")
async def retriever_status():
    """
    Check if the retriever functions are available
    """
    try:
        # Test if functions are importable
        from core.retriever import retrieve_nodes, query_index
        return {"status": "available", "functions": ["retrieve_nodes", "query_index", "indexes"]}
    except ImportError as e:
        return {"status": "unavailable", "error": str(e)}