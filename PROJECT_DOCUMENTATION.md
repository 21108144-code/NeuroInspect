================================================================================
                    NEUROINSPECT - PROJECT DOCUMENTATION
            AI-Powered Industrial Defect Detection System
                    Version 1.0.0 | December 2025
================================================================================


TABLE OF CONTENTS
--------------------------------------------------------------------------------
1. Project Overview
2. System Architecture  
3. Technology Stack
4. Installation & Setup
5. Detection Pipeline
6. API Reference
7. Frontend Pages
8. Configuration
9. Testing
10. Performance
11. Future Improvements


================================================================================
1. PROJECT OVERVIEW
================================================================================

What is NeuroInspect?
---------------------
NeuroInspect is an AI-powered industrial quality control system that 
automatically detects, localizes, and classifies defects in manufactured 
products using computer vision and deep learning techniques.


Problem Statement
-----------------
In manufacturing and quality assurance:
- Manual inspection is slow, expensive, and prone to human error
- Inconsistent results due to inspector fatigue and subjectivity
- No quantitative data for trend analysis and process improvement
- Bottleneck in production lines due to inspection time


Solution
--------
NeuroInspect provides:
- Real-time detection (100-200ms per image)
- Consistent, objective results 
- Quantified metrics (severity scores, area measurements)
- Analytics & trends for continuous improvement
- Explainable AI visualizations


Supported Defect Types
----------------------
Defect Type     Description                     Detection Method
---------------------------------------------------------------------------
Crack           Linear fractures in material    Vertical edge detection
Scratch         Surface abrasions               Horizontal edge detection
Stain           Contamination, discoloration    Large area blob detection
Dent            Surface depressions             Compact region detection
Corrosion       Material degradation            High-intensity large areas
Hole            Perforations, pits              Circular region detection
Anomaly         Unclassified defects            Fallback category


================================================================================
2. SYSTEM ARCHITECTURE
================================================================================

High-Level Architecture
-----------------------

+-------------------------------------------------------------------------+
|                            CLIENT LAYER                                  |
|  +-------------------------------------------------------------------+  |
|  |                  React Frontend (Port 5173)                        |  |
|  |  Dashboard | Inspection | Analytics | Root Cause | XAI | Settings |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
                                  |
                                  | HTTP/REST API
                                  v
+-------------------------------------------------------------------------+
|                              API LAYER                                   |
|  +-------------------------------------------------------------------+  |
|  |                 FastAPI Backend (Port 8080)                        |  |
|  |  /inspect | /defects | /root_cause | /xai | /health | /settings   |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------------+
|                         PROCESSING LAYER                                 |
|  +---------------+  +---------------+  +---------------+                |
|  |   Detector    |  |   Localizer   |  | Severity      |                |
|  | (ResNet18 +   |  | (Connected    |  | Scorer        |                |
|  |  Thresholding)|  |  Components)  |  |               |                |
|  +---------------+  +---------------+  +---------------+                |
|  +---------------+  +---------------+  +---------------+                |
|  | Root Cause    |  | Explainability|  |   Database    |                |
|  | Analyzer      |  | Engine        |  |   Manager     |                |
|  +---------------+  +---------------+  +---------------+                |
+-------------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------------+
|                            DATA LAYER                                    |
|  +-------------------------------------------------------------------+  |
|  |                  SQLite Database (Async)                           |  |
|  |            Inspections | Defects | Analyses                        |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+


Directory Structure
-------------------
NeuroInspect/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── inspect.py      (Image inspection endpoints)
│   │   │   │   ├── defects.py      (Defect query endpoints)
│   │   │   │   ├── root_cause.py   (Root cause analysis)
│   │   │   │   └── xai.py          (Explainability endpoints)
│   │   │   └── deps.py             (Dependency injection)
│   │   ├── cv/
│   │   │   ├── pretrained_detector.py  (Main detection logic)
│   │   │   ├── localizer.py        (Defect localization)
│   │   │   ├── severity.py         (Severity scoring)
│   │   │   ├── clustering.py       (Root cause clustering)
│   │   │   └── explainability.py   (Grad-CAM, attention maps)
│   │   ├── models/
│   │   │   ├── database.py         (SQLAlchemy models)
│   │   │   └── schemas.py          (Pydantic schemas)
│   │   ├── utils/
│   │   │   └── image_utils.py      (Image processing utilities)
│   │   ├── config.py               (Configuration settings)
│   │   └── main.py                 (FastAPI application)
│   ├── data/                       (SQLite database storage)
│   ├── test_images/                (Sample test images)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Header.tsx      (Top navigation bar)
│   │   │   │   ├── Sidebar.tsx     (Side navigation)
│   │   │   │   └── ThemeProvider.tsx
│   │   │   └── ui/                 (ShadCN UI components)
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx       (KPIs and charts)
│   │   │   ├── Inspection.tsx      (Image upload & analysis)
│   │   │   ├── Analytics.tsx       (Historical data)
│   │   │   ├── RootCause.tsx       (Pattern analysis)
│   │   │   ├── Explainability.tsx  (AI explanations)
│   │   │   └── Settings.tsx        (Configuration)
│   │   ├── lib/
│   │   │   └── api.ts              (Axios API client)
│   │   └── App.tsx                 (Main app component)
│   ├── package.json
│   ├── vite.config.ts
│   └── Dockerfile
├── docker-compose.yml
└── README.md


================================================================================
3. TECHNOLOGY STACK
================================================================================

Backend
-------
Technology      Version     Purpose
---------------------------------------------------------------------------
Python          3.10+       Programming language
FastAPI         0.100+      REST API framework
Uvicorn         Latest      ASGI server
PyTorch         2.0+        Deep learning framework
torchvision     0.15+       Pre-trained models (ResNet18)
OpenCV          4.8+        Image processing
SQLAlchemy      2.0+        Async ORM
Pydantic        2.0+        Data validation
Loguru          Latest      Logging


Frontend
--------
Technology      Version     Purpose
---------------------------------------------------------------------------
React           18.x        UI framework
TypeScript      5.x         Type-safe JavaScript
Vite            5.x         Build tool
Tailwind CSS    3.x         Styling
ShadCN/UI       Latest      Component library
Recharts        2.x         Charts and visualizations
Axios           1.x         HTTP client
React Router    6.x         Client-side routing


DevOps
------
Technology      Purpose
---------------------------------------------------------------------------
Docker          Containerization
docker-compose  Multi-container orchestration
Nginx           Production frontend serving


================================================================================
4. INSTALLATION & SETUP
================================================================================

Prerequisites
-------------
- Python 3.10+
- Node.js 18+
- NVIDIA GPU (optional, for faster inference)
- CUDA Toolkit (optional, for GPU support)


Option 1: Local Development
---------------------------

Backend Setup:

    cd c:\NeuroInspect\backend
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    mkdir data
    uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload


Frontend Setup:

    cd c:\NeuroInspect\frontend
    npm install
    npm run dev


Option 2: Docker
----------------

    cd c:\NeuroInspect
    docker-compose up --build

    Access:
    - Frontend: http://localhost:4000
    - Backend: http://localhost:8080
    - API Docs: http://localhost:8080/docs


================================================================================
5. DETECTION PIPELINE
================================================================================

Overview
--------
Input Image -> Pre-processing -> Detection -> Localization -> Classification -> Severity -> Output


Step 1: Pre-processing
----------------------
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)


Step 2: Detection (Dark-Line Thresholding)
------------------------------------------
The detector identifies defects as dark regions against a lighter background:

    # Statistical thresholding
    mean_val = np.mean(blur)
    std_val = np.std(blur)
    dark_threshold = mean_val - 1.5 * std_val
    
    # Create binary mask of dark pixels
    dark_mask = (blur < dark_threshold).astype(np.uint8) * 255
    
    # Adaptive thresholding as backup
    adaptive_mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # Combine both approaches
    combined_mask = cv2.bitwise_or(dark_mask, adaptive_mask)


Step 3: Localization (Connected Components)
-------------------------------------------
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    # Keep only the largest component (main defect)
    component_areas = [(i, stats[i][4]) for i in range(1, num_labels)]
    component_areas.sort(key=lambda x: x[1], reverse=True)
    largest_component = component_areas[0][0]
    
    # Extract bounding box
    x, y, w, h, area = stats[largest_component]


Step 4: Classification
----------------------
    def classify_defect(aspect_ratio, area_pct, intensity):
        if intensity > 0.5:
            return DefectType.CRACK
        elif aspect_ratio > 4:
            return DefectType.SCRATCH  # Horizontal
        elif aspect_ratio < 0.25:
            return DefectType.CRACK    # Vertical
        elif area_pct > 5:
            return DefectType.STAIN if intensity < 0.4 else DefectType.CORROSION
        elif 0.5 < aspect_ratio < 2.0 and area_pct > 0.5:
            return DefectType.DENT
        else:
            return DefectType.CRACK  # Default


Step 5: Severity Scoring
------------------------
    severity_score = (
        weight_area * area_percentage +
        weight_intensity * max_intensity +
        weight_location * location_score
    ) / total_weight
    
    # Map to severity level
    if severity_score < 0.33: return SeverityLevel.LOW
    elif severity_score < 0.66: return SeverityLevel.MEDIUM
    else: return SeverityLevel.HIGH


================================================================================
6. API REFERENCE
================================================================================

Base URL: http://localhost:8080


POST /inspect
-------------
Upload and analyze an image for defects.

Request:
    curl -X POST "http://localhost:8080/inspect" \
      -F "file=@image.png" \
      -F "confidence_threshold=0.5" \
      -F "return_heatmap=true" \
      -F "return_mask=true"

Response:
    {
      "inspection_id": "uuid",
      "timestamp": "2024-12-29T12:00:00Z",
      "image_name": "image.png",
      "image_width": 256,
      "image_height": 256,
      "processing_time_ms": 113,
      "is_defective": true,
      "overall_score": 0.659,
      "defects": [
        {
          "id": "abc123",
          "defect_type": "crack",
          "confidence": 0.179,
          "severity": "high",
          "severity_score": 0.659,
          "bounding_box": {
            "x_min": 0.55,
            "y_min": 0.08,
            "x_max": 0.72,
            "y_max": 0.98
          },
          "area_percentage": 9.36,
          "pixel_count": 6144
        }
      ],
      "heatmap_base64": "...",
      "mask_base64": "..."
    }


GET /health
-----------
Check system health and GPU status.

Response:
    {
      "status": "healthy",
      "version": "1.0.0",
      "model_loaded": true,
      "device": "cuda",
      "gpu_available": true,
      "gpu_name": "NVIDIA GeForce RTX 2050"
    }


GET /defects
------------
Query historical defect records.


POST /root_cause/analyze
------------------------
Perform clustering analysis on defect patterns.


POST /xai/explain
-----------------
Generate explainability visualizations (Grad-CAM).


================================================================================
7. FRONTEND PAGES
================================================================================

1. Dashboard
------------
   - KPI Cards: Total inspections, defect rate, average processing time
   - Trend Chart: Defect counts over time
   - Severity Distribution: Pie chart of low/medium/high
   - Defect Types: Bar chart of defect categories


2. Inspection
-------------
   - Image Upload: Drag & drop interface
   - View Modes: Original, Heatmap, Mask
   - Bounding Boxes: Visual defect localization
   - Defect Cards: Detailed information per defect


3. Analytics
------------
   - Time Range Filters: Hourly, daily, weekly
   - Data Table: Paginated inspection history
   - Summary Statistics: Aggregated metrics


4. Root Cause Analysis
----------------------
   - Clustering Parameters: Time range, cluster size
   - Cluster Cards: Grouped patterns
   - Recommendations: Suggested actions


5. Explainability
-----------------
   - Method Selection: Grad-CAM, Occlusion, Attention
   - Layer Selection: Which model layer to visualize
   - Side-by-side: Original vs. explanation view


6. Settings
-----------
   - Detection Thresholds: Adjust sensitivity
   - Severity Weights: Customize scoring
   - System Status: Health indicators


================================================================================
8. CONFIGURATION
================================================================================

Backend Configuration (app/config.py)
-------------------------------------
    class Settings(BaseSettings):
        # API Settings
        api_port: int = 8000
        cors_origins: str = "http://localhost:3000,http://localhost:5173"
        
        # Model Settings
        model_device: str = "cuda"  # or "cpu"
        detection_threshold: float = 0.5
        localization_threshold: float = 0.3
        
        # Severity Weights
        severity_weights_area: float = 0.4
        severity_weights_intensity: float = 0.35
        severity_weights_location: float = 0.25
        
        # Database
        database_url: str = "sqlite+aiosqlite:///./data/neuroinspect.db"


Environment Variables
---------------------
Variable                Default         Description
---------------------------------------------------------------------------
MODEL_DEVICE            cuda            Inference device (cuda/cpu)
DETECTION_THRESHOLD     0.5             Defect detection sensitivity
CORS_ORIGINS            *               Allowed CORS origins
DATABASE_URL            sqlite:///...   Database connection URL


================================================================================
9. TESTING
================================================================================

Test Images Location
--------------------
c:\NeuroInspect\backend\test_images\


Available Test Files
--------------------
File                            Expected Detection
---------------------------------------------------------------------------
normal_metal_1.png              No defects
defect_crack_heavy.png          Crack (high severity)
defect_scratch_heavy.png        Scratch
defect_stain_heavy.png          Stain
defect_hole_heavy.png           Hole
defect_mixed_*.png              Multiple defect types


Running Tests
-------------
    # Generate test images
    cd c:\NeuroInspect\backend
    python generate_test_images.py
    
    # Test API endpoint
    curl -X POST "http://localhost:8080/inspect" \
      -F "file=@test_images/defect_crack_heavy.png"


================================================================================
10. PERFORMANCE
================================================================================

Benchmarks (RTX 2050)
---------------------
Metric                  Value
---------------------------------------------------------------------------
Inference Time          ~100-200ms
Images per Second       ~5-10
GPU Memory              ~500MB
CPU Fallback            ~500-800ms


Optimization Tips
-----------------
1. Use GPU: 3-5x faster than CPU
2. Batch Processing: Use /inspect/batch for multiple images
3. Disable Heatmap: Set return_heatmap=false if not needed
4. Lower Resolution: Resize images to 512x512 or smaller


================================================================================
11. FUTURE IMPROVEMENTS
================================================================================

Planned Features
----------------
[ ] Video Stream Support: Real-time inspection from cameras
[ ] Model Training: Fine-tune on custom defect datasets
[ ] Multi-GPU: Distributed inference
[ ] Cloud Deployment: AWS/Azure/GCP templates
[ ] Mobile App: React Native companion app
[ ] Webhook Alerts: Notify on critical defects
[ ] Export Reports: PDF/Excel export


Model Improvements
------------------
[ ] Train on MVTec AD dataset for better accuracy
[ ] Add object detection (YOLO) for specific defect types
[ ] Implement ensemble models for robustness


================================================================================
LICENSE
================================================================================
MIT License - See LICENSE file for details.


================================================================================
AUTHOR
================================================================================
Abdul Mueed Kakar
AI/ML Engineer
December 2024
================================================================================
