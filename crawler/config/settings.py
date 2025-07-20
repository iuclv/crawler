"""Configuration management for the Image Similarity Crawler."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        # Project root directory
        self.PROJECT_ROOT = Path(__file__).parent.parent
        
        # API Configuration
        self.SERPAPI_KEY: Optional[str] = os.getenv('SERPAPI_KEY')
        
        # Search Settings
        self.MAX_RESULTS_PER_IMAGE: int = int(os.getenv('MAX_RESULTS_PER_IMAGE', '50'))
        self.SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.8'))
        self.CONCURRENT_DOWNLOADS: int = int(os.getenv('CONCURRENT_DOWNLOADS', '10'))
        
        # Storage Paths
        self.OUTPUT_DIRECTORY: Path = Path(os.getenv('OUTPUT_DIRECTORY', './data/output'))
        self.CACHE_DIRECTORY: Path = Path(os.getenv('CACHE_DIRECTORY', './data/cache'))
        self.MAX_CACHE_SIZE_GB: int = int(os.getenv('MAX_CACHE_SIZE_GB', '5'))
        
        # API Rate Limiting
        self.SERPAPI_REQUESTS_PER_HOUR: int = int(os.getenv('SERPAPI_REQUESTS_PER_HOUR', '100'))
        self.MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))
        self.TIMEOUT_SECONDS: int = int(os.getenv('TIMEOUT_SECONDS', '30'))
        
        # Logging Configuration
        self.LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE: Path = Path(os.getenv('LOG_FILE', './logs/crawler.log'))
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.OUTPUT_DIRECTORY,
            self.CACHE_DIRECTORY,
            self.LOG_FILE.parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate that all required settings are present."""
        if not self.SERPAPI_KEY:
            print("Warning: SERPAPI_KEY not set. SerpAPI functionality will be disabled.")
            return False
        return True

# Global settings instance
settings = Settings()
