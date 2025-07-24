import os
from typing import List

class Settings:
    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    API_RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    
    # Environment: 'production' or 'development'
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["https://wisdomofcrowd.net", 
                                  "http://wisdomofcrowd.net",
                                  "http://localhost:3000",
                                  "http://localhost:3001",
                                  "http://127.0.0.1:3000",
                                  "http://127.0.0.1:3001",
                                  ]  # Configure for production
    
    # Application Settings
    APP_NAME: str = "Documentation Agent API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # Retrieval Settings
    DEFAULT_TOP_K: int = 5
    DEFAULT_MAX_TOKENS: int = 512
    DEFAULT_TEMPERATURE: float = 0.1

settings = Settings()