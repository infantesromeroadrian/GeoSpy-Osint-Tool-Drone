# Quick Start Guide - Drone OSINT GeoSpy

This guide will help you get up and running with Drone OSINT GeoSpy quickly.

## 1. Prerequisites

Make sure you have:
- Python 3.9 or newer installed
- An OpenAI API key with GPT-4V access
- (Optional) A Google Maps API key for enhanced geolocation

## 2. Installation

```bash
# Clone the repository (or download and extract the ZIP file)
git clone https://github.com/your-org/Drone-Osint-GeoSpy.git
cd Drone-Osint-GeoSpy

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Configuration

1. Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

2. Make sure the required directories exist:
```bash
mkdir -p data/uploads data/frames data/results
```

## 4. Running the Application

### Option A: Run Backend and Frontend Together

For the simplest approach:

```bash
# Terminal 1
python src/main.py

# Terminal 2
python src/run_frontend.py
```

Then visit: http://localhost:8501 in your browser

### Option B: Run Individual Components

```bash
# Run Backend API Server
uvicorn src.backend.api:app --reload --host 0.0.0.0 --port 8000

# Run Frontend
streamlit run src/frontend/app.py
```

## 5. Using the Application

### Analyzing Static Images
1. Select the "IMAGE ANALYSIS" tab
2. Upload an image using the file uploader
3. Click "ANALYZE TERRAIN"
4. View the analysis results, including:
   - Map with exact coordinates
   - Location details (country, city, district, etc.)
   - Confidence assessment
   - Intelligence markers and reasoning

### Connecting to Drone Streams
1. Select the "DRONE STREAM" tab
2. Enter the RTSP/HTTP URL of your drone stream
3. Click "CONNECT TO DRONE FEED"
4. Use "CAPTURE & ANALYZE FRAME" to analyze the current view

### Chatting with the Vision LLM
1. Upload an image or connect to a stream first
2. Select the "INTERROGATION" tab
3. Enter your questions about the terrain or features visible in the image
4. The Vision LLM will respond with detailed information

## 6. Sample Images

If you don't have drone imagery, you can test with:
- Sample images in the `data/test_images` directory (if available)
- Satellite imagery from Google Earth
- Aerial photos from stock photography sites

## 7. Troubleshooting

- If you encounter OpenAI API errors, verify your API key and quota
- For stream connection issues, check that your stream URL is accessible
- If the frontend can't connect to the backend, ensure the backend server is running on http://localhost:8000

## 8. Next Steps

- Explore the API documentation at http://localhost:8000/docs
- Check the full README.md for more details
- Review the code to understand the architecture 