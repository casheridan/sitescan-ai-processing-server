import os
import cv2
import numpy as np
import aiofiles
import httpx
from typing import List
import tempfile
from urllib.parse import urlparse

class VideoProcessor:
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    async def download_video(self, video_url: str, job_id: str) -> str:
        """Download video from URL to local storage"""
        try:
            # Create temporary file
            video_path = os.path.join(self.temp_dir, f"{job_id}_video.mp4")
            
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", video_url) as response:
                    response.raise_for_status()
                    
                    async with aiofiles.open(video_path, 'wb') as f:
                        async for chunk in response.aiter_bytes():
                            await f.write(chunk)
            
            print(f"Video downloaded successfully: {video_path}")
            return video_path
            
        except Exception as e:
            print(f"Error downloading video: {e}")
            raise
    
    async def extract_frames(self, video_path: str, job_id: str, frame_interval: float = 1.0) -> List[np.ndarray]:
        """Extract frames from video at specified intervals"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"Video properties: {fps} FPS, {total_frames} frames, {duration:.2f}s duration")
            
            # Calculate frame interval
            frame_skip = int(fps * frame_interval)
            if frame_skip < 1:
                frame_skip = 1
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at intervals
                if frame_count % frame_skip == 0:
                    # Preprocess frame
                    processed_frame = self.preprocess_frame(frame)
                    frames.append(processed_frame)
                
                frame_count += 1
            
            cap.release()
            
            print(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for analysis"""
        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Normalize pixel values
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        return frame_normalized
    
    async def save_processed_video(self, frames: List[np.ndarray], job_id: str, fps: int = 30) -> str:
        """Save processed frames as video"""
        try:
            output_path = os.path.join(self.temp_dir, f"{job_id}_processed.mp4")
            
            if not frames:
                raise ValueError("No frames to save")
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert back to BGR for OpenCV
                if len(frame.shape) == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                # Convert to uint8
                frame_uint8 = (frame_bgr * 255).astype(np.uint8)
                out.write(frame_uint8)
            
            out.release()
            
            print(f"Processed video saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error saving processed video: {e}")
            raise
    
    def cleanup_temp_files(self, job_id: str):
        """Clean up temporary files for a job"""
        try:
            patterns = [
                f"{job_id}_video.mp4",
                f"{job_id}_processed.mp4",
                f"{job_id}_frames_*"
            ]
            
            for pattern in patterns:
                file_path = os.path.join(self.temp_dir, pattern)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up: {file_path}")
                    
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video information"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
            }
            
            cap.release()
            return info
            
        except Exception as e:
            print(f"Error getting video info: {e}")
            raise 