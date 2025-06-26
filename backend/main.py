
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))  # add project root to path
from backend.routes import retriever

# Create FastAPI app instance
app = FastAPI(
    title="Documentation Agent API",
    description="API for documentation agent with retrieval capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(retriever.router, prefix="/api/v1", tags=["retriever"])

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Documentation Agent API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # prod

    # uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000  # dev for hot reload
