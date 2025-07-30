from flask import Flask, request, jsonify
import os
import boto3
import tempfile
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
import time
import httpx
from typing import List, Dict, Any

app = Flask(__name__)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
)

# Global variable to store the loaded model
yolo_model = None
model_path = None

def download_model_from_s3():
    """Download YOLO model from S3"""
    try:
        bucket_name = os.environ.get('AWS_S3_BUCKET')
        model_key = os.environ.get('MODEL_S3_KEY', 'models/model-0.0.1.pt')
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        temp_path = temp_file.name
        temp_file.close()
        
        # Download from S3
        s3_client.download_file(bucket_name, model_key, temp_path)
        print(f"Model downloaded from s3://{bucket_name}/{model_key}")
        
        return temp_path
    except Exception as e:
        print(f"Error downloading model from S3: {e}")
        return None

def load_yolo_model():
    """Load YOLO model from S3"""
    global yolo_model, model_path
    
    if yolo_model is None:
        print("Loading YOLO model from S3...")
        model_path = download_model_from_s3()
        
        if model_path:
            try:
                yolo_model = YOLO(model_path)
                print("YOLO model loaded successfully!")
                return True
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                return False
        else:
            print("Failed to download model from S3")
            return False
    
    return True

def determine_severity(confidence: float) -> str:
    """Determine severity based on confidence"""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    else:
        return "low"

def get_recommendations(defect_type: str) -> List[str]:
    """Get recommendations based on defect type"""
    recommendations = {
        "crack": ["Inspect for structural damage", "Consider professional assessment"],
        "corrosion": ["Clean affected area", "Apply protective coating"],
        "leak": ["Identify source", "Repair immediately"],
        "damage": ["Document extent", "Plan repairs"],
        "default": ["Monitor condition", "Schedule maintenance"]
    }
    return recommendations.get(defect_type.lower(), recommendations["default"])

def analyze_video_with_yolo(video_path: str) -> Dict[str, Any]:
    """Analyze video using YOLO model"""
    if not load_yolo_model():
        raise Exception("Failed to load YOLO model")
    
    start_time = time.time()
    defects = []
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection every 10 frames (adjust as needed)
        if frame_count % 10 == 0:
            results = yolo_model(frame, conf=0.5)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = yolo_model.names[class_id]
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        defect = {
                            "timestamp": frame_count / fps if fps > 0 else frame_count / 30,
                            "type": class_name,
                            "confidence": confidence,
                            "location": {
                                "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                                "bbox": {
                                    "x": float(x1),
                                    "y": float(y1),
                                    "width": float(x2 - x1),
                                    "height": float(y2 - y1)
                                },
                                "frame_position": f"({int(x1)}, {int(y1)})"
                            },
                            "severity": determine_severity(confidence),
                            "description": f"Detected {class_name} with {confidence:.2f} confidence",
                            "recommendations": get_recommendations(class_name),
                            "frame_info": {
                                "frame_number": frame_count,
                                "frame_time": frame_count / fps if fps > 0 else frame_count / 30,
                                "video_fps": fps
                            }
                        }
                        defects.append(defect)
        
        frame_count += 1
    
    cap.release()
    
    # Generate summary
    total_defects = len(defects)
    critical_issues = len([d for d in defects if d["severity"] == "high"])
    defect_types = {}
    for defect in defects:
        defect_type = defect["type"]
        defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
    
    processing_time = time.time() - start_time
    
    return {
        "defects": defects,
        "summary": {
            "totalDefects": total_defects,
            "criticalIssues": critical_issues,
            "defectTypes": defect_types,
            "recommendedActions": f"Found {total_defects} defects. {critical_issues} critical issues require immediate attention."
        },
        "processing_time": processing_time
    }

@app.route('/health')
def health():
    model_loaded = yolo_model is not None
    return jsonify({
        "status": "healthy",
        "message": "AI server deployed successfully",
        "s3_bucket": os.environ.get('AWS_S3_BUCKET'),
        "model_loaded": model_loaded,
        "model_path": model_path or "not_loaded"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded video file"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        
        # Save uploaded file temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_file.save(temp_video.name)
        temp_video.close()
        
        try:
            # Analyze video
            result = analyze_video_with_yolo(temp_video.name)
            return jsonify(result)
        finally:
            # Clean up temporary file
            os.unlink(temp_video.name)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze video from URL"""
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        
        if not video_url:
            return jsonify({"error": "video_url is required"}), 400
        
        # Download video from URL
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        
        async def download_video():
            async with httpx.AsyncClient() as client:
                response = await client.get(video_url)
                response.raise_for_status()
                temp_video.write(response.content)
                temp_video.close()
        
        # For now, use synchronous download
        import requests
        response = requests.get(video_url)
        response.raise_for_status()
        temp_video.write(response.content)
        temp_video.close()
        
        try:
            # Analyze video
            result = analyze_video_with_yolo(temp_video.name)
            return jsonify(result)
        finally:
            # Clean up temporary file
            os.unlink(temp_video.name)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model-info')
def model_info():
    model_loaded = yolo_model is not None
    return jsonify({
        "model_path": "s3://" + os.environ.get('AWS_S3_BUCKET', 'not-set') + "/models/",
        "confidence_threshold": 0.5,
        "device": "cpu",
        "model_loaded": model_loaded,
        "message": "Model loaded from S3" if model_loaded else "Model will be loaded from S3 on first request"
    })

@app.route('/test-s3')
def test_s3():
    """Test S3 connection"""
    try:
        bucket_name = os.environ.get('AWS_S3_BUCKET')
        if not bucket_name:
            return jsonify({"error": "AWS_S3_BUCKET not set"}), 400
        
        # List objects in bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
        objects = [obj['Key'] for obj in response.get('Contents', [])]
        
        return jsonify({
            "status": "success",
            "bucket": bucket_name,
            "objects": objects,
            "message": "S3 connection working"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "S3 connection failed"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 