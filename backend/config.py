import os
from typing import List

class Settings:
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["*"]  # Configure for production
    
    # Application Settings
    APP_NAME: str = "Documentation Agent API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Retrieval Settings
    DEFAULT_TOP_K: int = 5
    DEFAULT_MAX_TOKENS: int = 512
    DEFAULT_TEMPERATURE: float = 0.1

settings = Settings()