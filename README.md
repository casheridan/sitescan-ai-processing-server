# AI Processing Server

This is a separate server dedicated to running YOLO models for video analysis. It receives jobs from the main sitescan-backend and processes videos to detect defects.

## Architecture

- **Language**: Python (better for ML libraries)
- **Framework**: FastAPI
- **Model**: YOLO (You Only Look Once) for object detection
- **Communication**: HTTP REST API with the main backend

## Features

- Video frame extraction and processing
- YOLO model inference
- Defect detection and classification
- Result formatting and storage
- Job status management
- Webhook notifications to main backend

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run the server:
```bash
python main.py
```

## API Endpoints

### POST /jobs
Start a new analysis job

### GET /jobs/{job_id}/status
Get job status

### POST /jobs/{job_id}/results
Update job results (internal use)

## Environment Variables

- `MAIN_BACKEND_URL`: URL of the main sitescan-backend
- `WEBHOOK_SECRET`: Secret for webhook signature verification
- `MODEL_PATH`: Path to YOLO model weights
- `GPU_ENABLED`: Whether to use GPU acceleration

## Model Configuration

Place your YOLO model weights in the `models/` directory and update the configuration in `config/model_config.py`. 