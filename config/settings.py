import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model settings
    model_path: str = "models/yolo_weights.pt"
    confidence_threshold: float = 0.5
    frame_interval: float = 1.0  # seconds between frames
    
    # Backend communication
    main_backend_url: str = "http://localhost:5000"
    webhook_secret: str = "your-webhook-secret-here"
    
    # Processing settings
    max_video_duration: int = 300  # 5 minutes
    temp_dir: str = "temp"
    
    # GPU settings
    gpu_enabled: bool = False
    device: str = "cpu"  # or "cuda"
    
    class Config:
        env_file = ".env"

_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings 