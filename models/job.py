from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Job:
    def __init__(
        self,
        id: str,
        video_id: str,
        user_id: str,
        video_url: str,
        status: JobStatus = JobStatus.PENDING
    ):
        self.id = id
        self.video_id = video_id
        self.user_id = user_id
        self.video_url = video_url
        self.status = status
        self.progress = 0
        self.error: Optional[str] = None
        self.results: Optional[Dict[str, Any]] = None
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None 