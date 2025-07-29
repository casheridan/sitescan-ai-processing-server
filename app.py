from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "message": "AI server deployed successfully"
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
        "model_path": "not_loaded_yet",
        "confidence_threshold": 0.5,
        "device": "cpu",
        "message": "Simple server - model not loaded yet"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 