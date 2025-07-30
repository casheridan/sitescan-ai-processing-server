from flask import Flask, request, jsonify
import os
import boto3
import tempfile
import shutil

app = Flask(__name__)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
)

def download_model_from_s3():
    """Download YOLO model from S3"""
    try:
        bucket_name = os.environ.get('AWS_S3_BUCKET')
        model_key = os.environ.get('MODEL_S3_KEY', 'models/best.pt')  # Default model path
        
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

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "message": "AI server deployed successfully",
        "s3_bucket": os.environ.get('AWS_S3_BUCKET'),
        "model_loaded": False  # Will be True once we implement model loading
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    return jsonify({
        "defects": [
            {
                "timestamp": 5.2,
                "type": "test_defect",
                "confidence": 0.85,
                "severity": "high",
                "description": "Test defect for deployment verification"
            }
        ],
        "summary": {
            "totalDefects": 1,
            "criticalIssues": 1,
            "recommendedActions": "Test deployment successful"
        },
        "processing_time": 1.0
    })

@app.route('/model-info')
def model_info():
    return jsonify({
        "model_path": "s3://" + os.environ.get('AWS_S3_BUCKET', 'not-set') + "/models/",
        "confidence_threshold": 0.5,
        "device": "cpu",
        "message": "Model will be loaded from S3 on first request"
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