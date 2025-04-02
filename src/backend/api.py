from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Query, Body
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import os
import shutil
import uuid
import json
from datetime import datetime
import re
import asyncio
import time
import cv2
import numpy as np
from PIL import Image
import io
import base64
import traceback
import logging
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import csv

# Import local modules - ensure compatibility with Docker and local environments
if "/app" in os.environ.get("PYTHONPATH", ""):
    # Docker environment
    from models.vision_llm import VisionLLM
    from utils.metadata_extractor import MetadataExtractor
    from utils.geo_service import GeoService
    from utils.video_processor import VideoProcessor
    from utils.mapbox_service import MapboxService
else:
    # Local environment
    from src.models.vision_llm import VisionLLM
    from src.utils.metadata_extractor import MetadataExtractor
    from src.utils.geo_service import GeoService
    from src.utils.video_processor import VideoProcessor
    from src.utils.mapbox_service import MapboxService

# Import the ObjectDetector class
try:
    from models.object_detector import ObjectDetector
except ImportError:
    try:
        from src.models.object_detector import ObjectDetector
    except ImportError:
        print("WARNING: Failed to import ObjectDetector")
        ObjectDetector = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple cache implementation
class SimpleCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key):
        if key in self.cache:
            # Update access time
            self.access_times[key] = time.time()
            logger.info(f"Cache hit for key: {key}")
            return self.cache[key]
        return None
    
    def set(self, key, value):
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            # Find least recently used entry
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            self.cache.pop(oldest_key)
            self.access_times.pop(oldest_key)
            logger.info(f"Cache eviction for key: {oldest_key}")
        
        # Add new entry
        self.cache[key] = value
        self.access_times[key] = time.time()
        logger.info(f"Cache set for key: {key}")
        
    def clear(self):
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")

# Create cache instances
map_cache = SimpleCache(max_size=50)  # Cache for maps
image_cache = SimpleCache(max_size=20)  # Cache for processed images
geocode_cache = SimpleCache(max_size=100)  # Cache for geocoding results

# Create FastAPI app
app = FastAPI(
    title="Drone OSINT GeoSpy API",
    description="API for processing drone imagery and video for geolocation analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class ChatRequest(BaseModel):
    image_id: str
    message: str

class AnalysisResponse(BaseModel):
    image_id: str
    metadata: Optional[Dict[str, Any]] = None
    llm_analysis: Optional[Dict[str, Any]] = None
    geo_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class MapRequest(BaseModel):
    latitude: float
    longitude: float
    zoom: int = 13

class GeocodeRequest(BaseModel):
    query: str
    limit: int = 5

class ReverseGeocodeRequest(BaseModel):
    latitude: float
    longitude: float

class ImageComparisonRequest(BaseModel):
    image_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class ObjectDetectionRequest(BaseModel):
    image_id: str
    confidence: float = 0.25
    model: str = "xlarge"

class InteractiveMapRequest(BaseModel):
    latitude: float
    longitude: float
    zoom: int = 13

class StaticMapRequest(BaseModel):
    latitude: float
    longitude: float
    style: str = "satellite-streets-v12"
    width: int = 800
    height: int = 600
    zoom: int = 15

class ComparisonRequest(BaseModel):
    image_id: str
    latitude: float
    longitude: float

class ExportRequest(BaseModel):
    image_id: str
    format: str = "json"  # "json", "csv", "pdf"

# Initialize services
metadata_extractor = MetadataExtractor()
geo_service = GeoService()
mapbox_service = MapboxService()

# Initialize VisionLLM
try:
    vision_llm = VisionLLM()
except Exception as e:
    print(f"WARNING: Failed to initialize VisionLLM: {str(e)}")
    vision_llm = None

video_processor = VideoProcessor(output_dir="./data/frames")

# Initialize the ObjectDetector
try:
    object_detector = ObjectDetector()
    print("Object Detector initialized successfully")
except Exception as e:
    print(f"WARNING: Failed to initialize ObjectDetector: {str(e)}")
    object_detector = None

# Ensure data directories exist
os.makedirs("./data/uploads", exist_ok=True)
os.makedirs("./data/frames", exist_ok=True)
os.makedirs("./data/results", exist_ok=True)

# Store active analysis sessions
active_sessions = {}

# Available YOLOv8 models
YOLO_MODELS = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt",
    "xlarge": "yolov8x.pt"
}

# Default model
current_model = "xlarge"
object_detector = ObjectDetector(model_name=YOLO_MODELS[current_model])

# Health check endpoint for Docker healthcheck
@app.get("/api/session/health", response_model=Dict[str, str])
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload/image", response_model=Dict[str, Any])
async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload an image for analysis.
    """
    try:
        # Generate a unique ID for this upload
        image_id = str(uuid.uuid4())
        
        # Create file path
        file_extension = os.path.splitext(file.filename)[1]
        upload_dir = "./data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{image_id}{file_extension}")
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create a new session for this image
        active_sessions[image_id] = {
            "image_id": image_id,
            "file_path": file_path,
            "upload_time": datetime.now().isoformat(),
            "filename": file.filename,
            "status": "uploaded"
        }
        
        # Start analysis in background if requested
        if background_tasks:
            background_tasks.add_task(analyze_image_task, image_id, file_path)
            return {
                "image_id": image_id,
                "status": "processing",
                "message": "Image uploaded successfully. Analysis started in background."
            }
        
        return {
            "image_id": image_id, 
            "file_path": file_path,
            "status": "uploaded",
            "message": "Image uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")

@app.post("/api/analyze/image/{image_id}", response_model=AnalysisResponse)
async def analyze_image(image_id: str, detect_objects: bool = False):
    """
    Analyze an uploaded image with Vision LLM and optionally detect objects.
    """
    try:
        # Check if the image exists in active sessions
        if image_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")
        
        session = active_sessions[image_id]
        file_path = session.get("file_path")
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Image file not found for ID {image_id}")
        
        # Extract metadata first
        metadata = metadata_extractor.extract_metadata(file_path)
        gps_coords = metadata.get("gps_coordinates")
        session["metadata"] = metadata
        
        # Analyze image with Vision LLM
        llm_analysis = vision_llm.analyze_image(file_path)
        session["llm_analysis"] = llm_analysis
        
        # Process geolocation data
        geo_data = None
        if llm_analysis:
            geo_coords = None
            geo_address = None
            
            # Check different possible structures
            if "location_assessment" in llm_analysis:
                # Extract coordinates from LLM analysis - older structure
                geo_coords = llm_analysis["location_assessment"].get("coordinates")
                geo_address = llm_analysis["location_assessment"].get("address")
            elif "geo_data" in llm_analysis:
                # New structure from Gemini
                geo_data = llm_analysis["geo_data"]
                if isinstance(geo_data, dict):
                    geo_coords = geo_data.get("coordinates")
            elif "llm_analysis" in llm_analysis and isinstance(llm_analysis["llm_analysis"], dict):
                # Nested structure
                if "location" in llm_analysis["llm_analysis"]:
                    location_data = llm_analysis["llm_analysis"]["location"]
                    if isinstance(location_data, dict):
                        geo_coords = location_data.get("coordinates")
                        geo_address = location_data
            
            # If we have direct geo_data from the structure
            if geo_data is None and isinstance(llm_analysis.get("geo_data"), dict):
                geo_data = llm_analysis["geo_data"]
            
            # If we still need to create geo_data from coordinates
            if geo_data is None and (geo_coords or geo_address):
                llm_geo_data = {
                    "coordinates": geo_coords,
                    "address": geo_address
                }
                
                # Merge LLM and metadata location data
                geo_data = geo_service.merge_location_data(llm_geo_data, gps_coords)
            elif geo_data is None:
                # Create a minimal geo_data structure
                geo_data = {
                    "sources": {},
                    "merged_data": {
                        "coordinates": {
                            "latitude": 0,
                            "longitude": 0
                        },
                        "address": {}
                    },
                    "confidence": {
                        "coordinates_source": "None"
                    }
                }
                
                # Still try to use metadata GPS if available
                if gps_coords:
                    geo_data["sources"]["metadata"] = {
                        "coordinates": {
                            "latitude": gps_coords.get("latitude"),
                            "longitude": gps_coords.get("longitude")
                        }
                    }
                    geo_data["merged_data"]["coordinates"] = geo_data["sources"]["metadata"]["coordinates"]
                    geo_data["confidence"]["coordinates_source"] = "Metadata"
            
            # Generate map if coordinates available
            map_html = None
            if geo_data and "merged_data" in geo_data and "coordinates" in geo_data["merged_data"]:
                coords = geo_data["merged_data"]["coordinates"]
                if coords and isinstance(coords, dict) and "latitude" in coords and "longitude" in coords:
                    try:
                        lat = coords["latitude"]
                        lon = coords["longitude"]
                        if lat != 0 and lon != 0:
                            map_html = geo_service.generate_map(float(lat), float(lon))
                    except Exception as map_error:
                        print(f"Error generating map: {str(map_error)}")
            
            if map_html:
                geo_data["map"] = map_html
        
        session["geo_data"] = geo_data
        
        # Optional: Perform object detection if requested
        object_detection_data = None
        if detect_objects and object_detector:
            try:
                detection_results = object_detector.detect_objects(file_path)
                
                if "error" not in detection_results or detection_results["annotated_image_path"] is not None:
                    session["object_detection"] = {
                        "timestamp": datetime.now().isoformat(),
                        "summary": detection_results["summary"],
                        "assessment": detection_results["assessment"],
                        "annotated_image_path": detection_results["annotated_image_path"]
                    }
                    session["has_object_detection"] = True
                    object_detection_data = detection_results
            except Exception as od_error:
                print(f"Error in object detection during analysis: {str(od_error)}")
                # Continue with analysis even if object detection fails
        
        # Compile the response
        response = {
            "image_id": image_id,
            "metadata": metadata,
            "llm_analysis": llm_analysis,
            "geo_data": geo_data
        }
        
        # Add object detection data if available
        if object_detection_data:
            response["object_detection"] = object_detection_data
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.post("/api/chat/image/{image_id}", response_model=Dict[str, Any])
async def chat_with_image(image_id: str, request: ChatRequest):
    """
    Chat with the Vision LLM about an image.
    """
    try:
        # Check if image exists in active sessions
        if image_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")
            
        session = active_sessions[image_id]
        file_path = session["file_path"]
        
        # Read image file as bytes
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Enviar directamente el mensaje a la función chat_about_image en lugar de analyze_image
        # ya que chat_about_image está diseñada para conversaciones
        response = vision_llm.chat_about_image(
            image_path=file_path,
            user_message=request.message
        )
        
        # Store conversation in session
        if "conversation" not in session:
            session["conversation"] = []
            
        session["conversation"].append({
            "role": "user",
            "message": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        session["conversation"].append({
            "role": "assistant",
            "message": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "image_id": image_id,
            "message": request.message,
            "response": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.post("/api/upload/video", response_model=Dict[str, Any])
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a video for analysis.
    """
    try:
        # Generate a unique ID for this upload
        video_id = str(uuid.uuid4())
        
        # Create file path
        file_extension = os.path.splitext(file.filename)[1]
        upload_dir = "./data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{video_id}{file_extension}")
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create a new session for this video
        active_sessions[video_id] = {
            "video_id": video_id,
            "file_path": file_path,
            "upload_time": datetime.now().isoformat(),
            "filename": file.filename,
            "status": "uploaded",
            "type": "video"
        }
        
        # Extract frames in background
        background_tasks.add_task(extract_video_frames_task, video_id, file_path)
        
        return {
            "video_id": video_id, 
            "file_path": file_path,
            "status": "processing",
            "message": "Video uploaded successfully. Frame extraction started."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading video: {str(e)}")

@app.post("/api/stream/connect", response_model=Dict[str, Any])
async def connect_stream(stream_url: str = Form(...)):
    """
    Connect to a video stream (like RTSP from a drone).
    """
    try:
        # Generate a unique ID for this stream
        stream_id = str(uuid.uuid4())
        
        # Create a new session for this stream
        active_sessions[stream_id] = {
            "stream_id": stream_id,
            "stream_url": stream_url,
            "connect_time": datetime.now().isoformat(),
            "status": "connecting",
            "type": "stream"
        }
        
        # Start processing the stream
        video_processor.start_stream_processing(stream_url)
        
        # Update session status
        active_sessions[stream_id]["status"] = "streaming"
        
        return {
            "stream_id": stream_id,
            "status": "streaming",
            "message": "Successfully connected to stream. Processing frames."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to stream: {str(e)}")

@app.post("/api/stream/disconnect/{stream_id}", response_model=Dict[str, Any])
async def disconnect_stream(stream_id: str):
    """
    Disconnect from a video stream.
    """
    try:
        # Check if stream exists in active sessions
        if stream_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Stream with ID {stream_id} not found")
            
        # Stop processing the stream
        video_processor.stop_stream_processing()
        
        # Update session status
        active_sessions[stream_id]["status"] = "disconnected"
        active_sessions[stream_id]["disconnect_time"] = datetime.now().isoformat()
        
        return {
            "stream_id": stream_id,
            "status": "disconnected",
            "message": "Successfully disconnected from stream."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error disconnecting from stream: {str(e)}")

@app.get("/api/stream/latest-frame/{stream_id}", response_model=Dict[str, Any])
async def get_latest_frame(stream_id: str, analyze: bool = False):
    """
    Get the latest frame from a video stream and optionally analyze it.
    """
    try:
        # Check if stream exists in active sessions
        if stream_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Stream with ID {stream_id} not found")
            
        # Get latest frame
        result = video_processor.get_latest_frame()
        if not result:
            return {
                "stream_id": stream_id,
                "status": "no_frame",
                "message": "No frames available yet."
            }
            
        frame, frame_path = result
        
        # Generate a unique ID for this frame
        frame_id = str(uuid.uuid4())
        
        # Store frame information
        active_sessions[frame_id] = {
            "frame_id": frame_id,
            "stream_id": stream_id,
            "file_path": frame_path,
            "capture_time": datetime.now().isoformat(),
            "status": "captured",
            "type": "frame"
        }
        
        response = {
            "frame_id": frame_id,
            "stream_id": stream_id,
            "status": "captured",
            "file_path": frame_path
        }
        
        # Analyze the frame if requested
        if analyze:
            # Extract metadata
            metadata = metadata_extractor.extract_metadata(frame_path)
            
            # Extract GPS coordinates from metadata if available
            gps_coords = metadata.get("gps_coordinates")
            
            # Analyze image with Vision LLM
            llm_analysis = vision_llm.analyze_image(frame_path)
            
            # Process geolocation data
            geo_data = None
            if "location_assessment" in llm_analysis:
                # Extract coordinates from LLM analysis
                coords = llm_analysis["location_assessment"].get("coordinates")
                address = llm_analysis["location_assessment"].get("address")
                
                llm_geo_data = {
                    "coordinates": coords,
                    "address": address
                }
                
                # Merge LLM and metadata location data
                geo_data = geo_service.merge_location_data(llm_geo_data, gps_coords)
            
            # Update frame information
            active_sessions[frame_id]["metadata"] = metadata
            active_sessions[frame_id]["llm_analysis"] = llm_analysis
            active_sessions[frame_id]["geo_data"] = geo_data
            active_sessions[frame_id]["status"] = "analyzed"
            
            # Add analysis to response
            response["status"] = "analyzed"
            response["metadata"] = metadata
            response["llm_analysis"] = llm_analysis
            response["geo_data"] = geo_data
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting latest frame: {str(e)}")

@app.get("/api/session/{session_id}", response_model=Dict[str, Any])
async def get_session(session_id: str):
    """
    Get information about an active session.
    """
    try:
        # Check if session exists
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Session with ID {session_id} not found")
            
        return active_sessions[session_id]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session: {str(e)}")

@app.post("/api/generate/map")
async def generate_map(request: MapRequest):
    """Generate a map for the given coordinates."""
    try:
        # Generate map using GeoService
        map_html = geo_service.generate_map(
            latitude=request.latitude,
            longitude=request.longitude,
            zoom=15
        )
        
        if not map_html:
            raise HTTPException(status_code=500, detail="Failed to generate map")
            
        return {"map_html": map_html}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating map: {str(e)}")

@app.post("/api/location/compare", response_model=Dict[str, Any])
async def compare_image_with_maps(request: ImageComparisonRequest):
    """
    Compare an uploaded image with maps using Mapbox.
    """
    try:
        # Check if image exists in active sessions
        if request.image_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Image with ID {request.image_id} not found")
            
        session = active_sessions[request.image_id]
        file_path = session.get("file_path")
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Image file not found for ID {request.image_id}")
        
        # Get coordinates from request or from session
        lat = request.latitude
        lon = request.longitude
        
        if lat is None or lon is None:
            # Try to get from session
            if "geo_data" in session and "merged_data" in session["geo_data"]:
                coords = session["geo_data"]["merged_data"].get("coordinates", {})
                lat = coords.get("latitude")
                lon = coords.get("longitude")
                
            if lat is None or lon is None:
                raise HTTPException(status_code=400, detail="No coordinates provided and none found in analysis")
        
        # Generate comparison HTML
        comparison_html = mapbox_service.generate_comparison_html(file_path, lat, lon)
        
        # Generate interactive map
        interactive_map = mapbox_service.generate_interactive_map(lat, lon)
        
        return {
            "image_id": request.image_id,
            "coordinates": {
                "latitude": lat,
                "longitude": lon
            },
            "comparison_html": comparison_html,
            "interactive_map": interactive_map
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing image with maps: {str(e)}")

@app.post("/api/location/satellite", response_model=Dict[str, Any])
async def get_satellite_image(request: MapRequest):
    """
    Get a satellite image for the given coordinates using Mapbox.
    """
    try:
        satellite_image = mapbox_service.get_satellite_image(
            latitude=request.latitude,
            longitude=request.longitude
        )
        
        if not satellite_image:
            raise HTTPException(status_code=500, detail="Failed to get satellite image")
        
        # Encode the image to base64 for response
        image_b64 = base64.b64encode(satellite_image).decode('utf-8')
        
        return {
            "coordinates": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "image_data": f"data:image/png;base64,{image_b64}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting satellite image: {str(e)}")

@app.post("/api/generate/interactive_map", response_model=Dict[str, Any])
async def generate_interactive_map(request: MapRequest):
    """
    Generate an interactive map for the given coordinates.
    """
    try:
        # Check cache first
        cache_key = f"map_{request.latitude}_{request.longitude}_{request.zoom}"
        cached_map = map_cache.get(cache_key)
        if cached_map:
            return cached_map
            
        map_html = geo_service.generate_interactive_map(request.latitude, request.longitude, zoom=request.zoom)
        
        if not map_html:
            raise HTTPException(status_code=500, detail="Failed to generate map")
            
        response = {"status": "success", "map_html": map_html}
        
        # Store in cache
        map_cache.set(cache_key, response)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating map: {str(e)}")

@app.post("/api/geocode/forward", response_model=Dict[str, Any])
async def geocode_forward(request: GeocodeRequest):
    """
    Forward geocoding - convert an address to coordinates.
    """
    try:
        # Check cache first
        cache_key = f"forward_{request.query}_{request.limit}"
        cached_result = geocode_cache.get(cache_key)
        if cached_result:
            return cached_result
            
        results = mapbox_service.geocode_forward(query=request.query, limit=request.limit)
        
        if not results:
            return {"status": "error", "message": "No results found"}
            
        response = {"status": "success", "results": results}
        
        # Store in cache
        geocode_cache.set(cache_key, response)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during forward geocoding: {str(e)}")

@app.post("/api/geocode/reverse", response_model=Dict[str, Any])
async def geocode_reverse(request: ReverseGeocodeRequest):
    """
    Reverse geocoding - convert coordinates to an address.
    """
    try:
        # Check cache first
        cache_key = f"reverse_{request.latitude}_{request.longitude}"
        cached_result = geocode_cache.get(cache_key)
        if cached_result:
            return cached_result
            
        address = mapbox_service.geocode_reverse(
            longitude=request.longitude, 
            latitude=request.latitude
        )
        
        if not address:
            return {"status": "error", "message": "Could not find address for these coordinates"}
            
        response = {"status": "success", "address": address}
        
        # Store in cache
        geocode_cache.set(cache_key, response)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during reverse geocoding: {str(e)}")

@app.post("/api/static-map", response_model=Dict[str, Any])
async def get_static_map(request: StaticMapRequest):
    """
    Generate a static map image for the given coordinates.
    """
    try:
        # Check cache first
        cache_key = f"static_{request.latitude}_{request.longitude}_{request.style}_{request.width}_{request.height}_{request.zoom}"
        cached_map = map_cache.get(cache_key)
        if cached_map:
            return cached_map
            
        # Get static map image from Mapbox
        map_image = mapbox_service.get_static_map(
            longitude=request.longitude,
            latitude=request.latitude,
            style=request.style,
            width=request.width,
            height=request.height,
            zoom=request.zoom
        )
        
        if map_image:
            # Convert to base64
            encoded_image = base64.b64encode(map_image).decode('utf-8')
            response = {
                "status": "success",
                "image_data": f"data:image/jpeg;base64,{encoded_image}"
            }
            
            # Store in cache
            map_cache.set(cache_key, response)
            
            return response
        else:
            raise HTTPException(status_code=500, detail="Failed to generate static map")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating static map: {str(e)}")

@app.post("/api/detect/objects", response_model=Dict[str, Any])
async def detect_objects(request: ObjectDetectionRequest):
    """
    Detect objects in an image using YOLOv8.
    """
    try:
        # Check if object detector is available
        if not object_detector:
            raise HTTPException(status_code=503, detail="Object detector not available")
            
        # Check if image exists in active sessions
        if request.image_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Image with ID {request.image_id} not found")
            
        session = active_sessions[request.image_id]
        file_path = session.get("file_path")
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Image file not found for ID {request.image_id}")
            
        # Perform object detection
        detection_results = object_detector.detect_objects(
            image_path=file_path,
            conf_threshold=request.confidence
        )
        
        # Check if detection was successful
        if "error" in detection_results and detection_results["annotated_image_path"] is None:
            raise HTTPException(status_code=500, detail=f"Object detection failed: {detection_results['error']}")
            
        # Store detection results in session
        session["object_detection"] = {
            "timestamp": datetime.now().isoformat(),
            "summary": detection_results["summary"],
            "assessment": detection_results["assessment"],
            "annotated_image_path": detection_results["annotated_image_path"]
        }
        
        # Update session status
        session["has_object_detection"] = True
        
        # Prepare response
        response = {
            "image_id": request.image_id,
            "detection_results": detection_results,
            "message": f"Detected {detection_results['summary']['total_objects_detected']} objects in image"
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in object detection: {str(e)}")

@app.get("/api/image/annotated/{image_id}")
async def get_annotated_image(image_id: str):
    """
    Get the annotated image with object detection bounding boxes.
    """
    try:
        # Check if image exists in active sessions
        if image_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")
            
        session = active_sessions[image_id]
        
        # Check if object detection has been performed
        if not session.get("has_object_detection", False):
            raise HTTPException(status_code=400, detail=f"No object detection performed for image {image_id}")
            
        # Get annotated image path
        annotated_img_path = session.get("object_detection", {}).get("annotated_image_path")
        
        if not annotated_img_path or not os.path.exists(annotated_img_path):
            # Try to find the image using the original file path pattern
            original_path = session.get("file_path", "")
            if original_path and os.path.exists(original_path):
                # Try different patterns for the annotated image
                possible_paths = [
                    original_path.replace(".", "_annotated."),
                    os.path.splitext(original_path)[0] + "_annotated" + os.path.splitext(original_path)[1],
                    os.path.join("./data/uploads", f"annotated_{os.path.basename(original_path)}")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        annotated_img_path = path
                        # Update the session with the correct path
                        if "object_detection" in session:
                            session["object_detection"]["annotated_image_path"] = path
                        break
            
            if not annotated_img_path or not os.path.exists(annotated_img_path):
                raise HTTPException(status_code=404, detail=f"Annotated image not found. Tried multiple paths: {possible_paths}")
            
        # Read and return the image
        with open(annotated_img_path, "rb") as img_file:
            img_data = img_file.read()
            
        # Convert to base64 for response
        img_b64 = base64.b64encode(img_data).decode('utf-8')
        
        return {"image_id": image_id, "annotated_image": f"data:image/jpeg;base64,{img_b64}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting annotated image: {str(e)}")

@app.post("/api/set/model")
async def set_detection_model(model: str = Body(..., description="Model type: nano, small, medium, large, xlarge")):
    """
    Set the YOLOv8 model to use for object detection.
    """
    global object_detector
    global current_model
    
    if model not in YOLO_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Available models: {', '.join(YOLO_MODELS.keys())}")
    
    if model != current_model:
        try:
            current_model = model
            # Initialize a new detector with the selected model
            object_detector = ObjectDetector(model_name=YOLO_MODELS[model])
            return {"status": "success", "message": f"Model changed to YOLOv8 {model}", "model": model}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error changing model: {str(e)}")
    else:
        return {"status": "success", "message": f"Already using YOLOv8 {model}", "model": model}

@app.get("/api/models")
async def get_available_models():
    """
    Get the list of available YOLOv8 models.
    """
    return {
        "available_models": list(YOLO_MODELS.keys()),
        "current_model": current_model,
        "models_info": {
            "nano": "Smallest and fastest model, lower accuracy",
            "small": "Small model with good balance of speed and accuracy",
            "medium": "Medium-sized model with better accuracy than small",
            "large": "Large model with good accuracy, slower processing",
            "xlarge": "Extra large model with best accuracy, slowest processing"
        }
    }

@app.post("/api/export/results")
async def export_analysis_results(request: ExportRequest):
    """
    Export analysis results in the requested format.
    """
    try:
        # Check if session exists
        if request.image_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"No analysis found for image ID: {request.image_id}")
        
        session = active_sessions[request.image_id]
        
        # Check if analysis exists
        if "analysis_results" not in session:
            raise HTTPException(status_code=404, detail=f"No analysis results found for image ID: {request.image_id}")
        
        analysis_results = session["analysis_results"]
        file_path = session.get("file_path", "")
        filename = os.path.basename(file_path) if file_path else request.image_id
        
        # Export based on format
        if request.format.lower() == "json":
            # JSON format
            json_data = json.dumps(analysis_results, indent=2)
            return JSONResponse(
                content=analysis_results,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}_analysis.json"
                }
            )
            
        elif request.format.lower() == "csv":
            # CSV format
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(["Category", "Key", "Value"])
            
            # LLM Analysis
            llm_analysis = analysis_results.get("llm_analysis", {})
            if isinstance(llm_analysis, dict):
                for key, value in llm_analysis.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            writer.writerow(["LLM Analysis", f"{key}.{subkey}", str(subvalue)])
                    else:
                        writer.writerow(["LLM Analysis", key, str(value)])
            else:
                writer.writerow(["LLM Analysis", "description", str(llm_analysis)])
            
            # Geo Data
            geo_data = analysis_results.get("geo_data", {})
            if isinstance(geo_data, dict):
                for key, value in geo_data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            writer.writerow(["Geo Data", f"{key}.{subkey}", str(subvalue)])
                    elif isinstance(value, list):
                        writer.writerow(["Geo Data", key, ", ".join(map(str, value))])
                    else:
                        writer.writerow(["Geo Data", key, str(value)])
            
            # Metadata
            metadata = analysis_results.get("metadata", {})
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            writer.writerow(["Metadata", f"{key}.{subkey}", str(subvalue)])
                    else:
                        writer.writerow(["Metadata", key, str(value)])
            
            # Object Detection
            obj_detection = analysis_results.get("object_detection", {})
            if isinstance(obj_detection, dict):
                obj_counts = obj_detection.get("object_counts", {})
                if obj_counts:
                    for obj, count in obj_counts.items():
                        writer.writerow(["Object Detection", obj, count])
                
                assessment = obj_detection.get("assessment", {})
                if assessment:
                    for key, value in assessment.items():
                        writer.writerow(["Detection Assessment", key, str(value)])
            
            csv_data = output.getvalue()
            return Response(
                content=csv_data,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}_analysis.csv"
                }
            )
            
        elif request.format.lower() == "pdf":
            # PDF format
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = styles["Heading1"]
            story.append(Paragraph(f"Analysis Report: {filename}", title_style))
            story.append(Spacer(1, 0.25*inch))
            
            # Add image if available
            if file_path and os.path.exists(file_path):
                try:
                    img_width = 6 * inch
                    img = RLImage(file_path, width=img_width, height=img_width*0.75)
                    story.append(img)
                    story.append(Spacer(1, 0.25*inch))
                except Exception as img_err:
                    story.append(Paragraph(f"Image could not be loaded: {str(img_err)}", styles["Normal"]))
            
            # LLM Analysis
            story.append(Paragraph("LLM Analysis", styles["Heading2"]))
            llm_analysis = analysis_results.get("llm_analysis", {})
            
            if isinstance(llm_analysis, dict):
                for key, value in llm_analysis.items():
                    if key == "description" or key == "analysis":
                        story.append(Paragraph(f"<b>{key.capitalize()}:</b> {value}", styles["Normal"]))
                        story.append(Spacer(1, 0.1*inch))
            else:
                story.append(Paragraph(f"<b>Description:</b> {llm_analysis}", styles["Normal"]))
            
            story.append(Spacer(1, 0.25*inch))
            
            # Geo Data
            geo_data = analysis_results.get("geo_data", {})
            if geo_data:
                story.append(Paragraph("Geolocation Data", styles["Heading2"]))
                
                location_data = []
                if isinstance(geo_data, dict):
                    # Location table
                    location_data.append(["Country", "City", "Neighborhood", "Street"])
                    location_data.append([
                        geo_data.get("country", "Unknown"),
                        geo_data.get("city", "Unknown"),
                        geo_data.get("neighborhood", "Unknown"),
                        geo_data.get("street", "Unknown")
                    ])
                    
                    # Create table
                    if len(location_data) > 1:
                        location_table = Table(location_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                        location_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(location_table)
                        story.append(Spacer(1, 0.2*inch))
                    
                    # Coordinates
                    coords = geo_data.get("coordinates", {})
                    if coords:
                        story.append(Paragraph(f"<b>Coordinates:</b> {coords.get('latitude', 'N/A')}, {coords.get('longitude', 'N/A')}", styles["Normal"]))
                        story.append(Spacer(1, 0.1*inch))
            
            # Object Detection
            obj_detection = analysis_results.get("object_detection", {})
            if obj_detection:
                story.append(Paragraph("Object Detection", styles["Heading2"]))
                
                # Object counts table
                obj_counts = obj_detection.get("object_counts", {})
                if obj_counts:
                    counts_data = [["Object Type", "Count"]]
                    for obj, count in obj_counts.items():
                        counts_data.append([obj.capitalize(), str(count)])
                    
                    counts_table = Table(counts_data, colWidths=[3*inch, 1*inch])
                    counts_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(counts_table)
                    story.append(Spacer(1, 0.2*inch))
                
                # Assessment
                assessment = obj_detection.get("assessment", {})
                if assessment:
                    story.append(Paragraph("<b>Assessment:</b>", styles["Normal"]))
                    for key, value in assessment.items():
                        story.append(Paragraph(f"<b>{key.replace('_', ' ').capitalize()}:</b> {value}", styles["Normal"]))
                
                # Add the annotated image if available
                annotated_img_path = obj_detection.get("annotated_image_path")
                if annotated_img_path and os.path.exists(annotated_img_path):
                    try:
                        story.append(Spacer(1, 0.25*inch))
                        story.append(Paragraph("Annotated Image", styles["Heading3"]))
                        img_width = 6 * inch
                        img = RLImage(annotated_img_path, width=img_width, height=img_width*0.75)
                        story.append(img)
                    except Exception as img_err:
                        story.append(Paragraph(f"Annotated image could not be loaded: {str(img_err)}", styles["Normal"]))
            
            # Build the PDF
            doc.build(story)
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return Response(
                content=pdf_data,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}_analysis.pdf"
                }
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {request.format}. Supported formats: json, csv, pdf")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error exporting results: {str(e)}")

# Background tasks
def analyze_image_task(image_id: str, file_path: str):
    """Background task to analyze an image."""
    try:
        # Update session status
        if image_id in active_sessions:
            active_sessions[image_id]["status"] = "analyzing"
        
        # Extract metadata
        metadata = metadata_extractor.extract_metadata(file_path)
        
        # Extract GPS coordinates from metadata if available
        gps_coords = metadata.get("gps_coordinates")
        
        # Analyze image with Vision LLM
        llm_analysis = vision_llm.analyze_image(file_path)
        
        # Process geolocation data
        geo_data = None
        if llm_analysis:
            geo_coords = None
            geo_address = None
            
            # Check different possible structures
            if "location_assessment" in llm_analysis:
                # Extract coordinates from LLM analysis - older structure
                geo_coords = llm_analysis["location_assessment"].get("coordinates")
                geo_address = llm_analysis["location_assessment"].get("address")
            elif "geo_data" in llm_analysis:
                # New structure from Gemini
                geo_data = llm_analysis["geo_data"]
                if isinstance(geo_data, dict):
                    geo_coords = geo_data.get("coordinates")
            elif "llm_analysis" in llm_analysis and isinstance(llm_analysis["llm_analysis"], dict):
                # Nested structure
                if "location" in llm_analysis["llm_analysis"]:
                    location_data = llm_analysis["llm_analysis"]["location"]
                    if isinstance(location_data, dict):
                        geo_coords = location_data.get("coordinates")
                        geo_address = location_data
            
            # If we have direct geo_data from the structure
            if geo_data is None and isinstance(llm_analysis.get("geo_data"), dict):
                geo_data = llm_analysis["geo_data"]
            
            # If we still need to create geo_data from coordinates
            if geo_data is None and (geo_coords or geo_address):
                llm_geo_data = {
                    "coordinates": geo_coords,
                    "address": geo_address
                }
                
                # Merge LLM and metadata location data
                geo_data = geo_service.merge_location_data(llm_geo_data, gps_coords)
            elif geo_data is None:
                # Create a minimal geo_data structure
                geo_data = {
                    "sources": {},
                    "merged_data": {
                        "coordinates": {
                            "latitude": 0,
                            "longitude": 0
                        },
                        "address": {}
                    },
                    "confidence": {
                        "coordinates_source": "None"
                    }
                }
                
                # Still try to use metadata GPS if available
                if gps_coords:
                    geo_data["sources"]["metadata"] = {
                        "coordinates": {
                            "latitude": gps_coords.get("latitude"),
                            "longitude": gps_coords.get("longitude")
                        }
                    }
                    geo_data["merged_data"]["coordinates"] = geo_data["sources"]["metadata"]["coordinates"]
                    geo_data["confidence"]["coordinates_source"] = "Metadata"
            
            # Generate map if coordinates available
            map_html = None
            if geo_data and "merged_data" in geo_data and "coordinates" in geo_data["merged_data"]:
                coords = geo_data["merged_data"]["coordinates"]
                if coords and isinstance(coords, dict) and "latitude" in coords and "longitude" in coords:
                    try:
                        lat = coords["latitude"]
                        lon = coords["longitude"]
                        if lat != 0 and lon != 0:
                            map_html = geo_service.generate_map(float(lat), float(lon))
                    except Exception as map_error:
                        print(f"Error generating map: {str(map_error)}")
            
            if map_html:
                geo_data["map"] = map_html
        
        # Update session with results
        if image_id in active_sessions:
            active_sessions[image_id]["metadata"] = metadata
            active_sessions[image_id]["llm_analysis"] = llm_analysis
            active_sessions[image_id]["geo_data"] = geo_data
            active_sessions[image_id]["status"] = "completed"
            
        # Save results to file
        results = {
            "image_id": image_id,
            "metadata": metadata,
            "llm_analysis": llm_analysis,
            "geo_data": geo_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Asegurarse de que llm_analysis tiene todos los campos necesarios
        if "analysis" in llm_analysis and "confidence_level" not in llm_analysis:
            # Asignar valores por defecto para los campos necesarios
            llm_analysis["confidence_level"] = {
                "overall": "medium",
                "country": "medium",
                "city": "medium",
                "district": "medium",
                "neighborhood": "medium",
                "street": "low",
                "coordinates": "medium"
            }
            
            # Si no tenemos evidence, extraer algo de la información
            if "evidence" not in llm_analysis:
                evidence = {
                    "landmarks": [],
                    "terrain_features": [],
                    "architectural_elements": [],
                    "signage": []
                }
                
                analysis_text = llm_analysis["analysis"]
                
                # Buscar menciones de edificios, monumentos, etc.
                if "edificios" in analysis_text.lower():
                    section = analysis_text.split("edificios", 1)[1].split("\n\n", 1)[0]
                    for line in section.split("\n"):
                        line = line.strip()
                        if line and not line.startswith("#"):
                            evidence["architectural_elements"].append(line)
                
                if "monumentos" in analysis_text.lower():
                    section = analysis_text.split("monumentos", 1)[1].split("\n\n", 1)[0]
                    for line in section.split("\n"):
                        line = line.strip()
                        if line and not line.startswith("#"):
                            evidence["landmarks"].append(line)
                
                # Añadir valores predeterminados si no se encontró nada
                if not evidence["landmarks"]:
                    evidence["landmarks"] = ["Edificios urbanos", "Skyline de la ciudad"]
                if not evidence["architectural_elements"]:
                    evidence["architectural_elements"] = ["Arquitectura moderna", "Edificios de gran altura"]
                if not evidence["terrain_features"]:
                    evidence["terrain_features"] = ["Área urbana", "Costa o litoral"]
                
                llm_analysis["evidence"] = evidence
            
            # Si no tenemos reasoning, crear uno genérico
            if "reasoning" not in llm_analysis:
                llm_analysis["reasoning"] = "Análisis basado en las características visuales de la imagen, incluyendo el estilo arquitectónico, entorno y otros elementos distintivos visibles en la imagen."
        
        # Asegurarse de que geo_data tiene todos los campos necesarios
        if geo_data:
            if "merged_data" not in geo_data:
                # Si no tenemos merged_data, pero tenemos text_analysis
                if "text_analysis" in geo_data:
                    # Crear valores por defecto
                    coordinates = {"latitude": None, "longitude": None}
                    address = {
                        "country": "Desconocido",
                        "city": "Desconocido",
                        "district": "Desconocido",
                        "neighborhood": "Desconocido",
                        "street": "Desconocido"
                    }
                    
                    # Tratar de extraer información del análisis
                    analysis_text = geo_data["text_analysis"]
                    if "país" in analysis_text.lower() or "country" in analysis_text.lower():
                        for line in analysis_text.split("\n"):
                            if "país" in line.lower() or "country" in line.lower():
                                parts = line.split(":")
                                if len(parts) > 1:
                                    address["country"] = parts[1].strip()
                                    break
                    
                    if "ciudad" in analysis_text.lower() or "city" in analysis_text.lower():
                        for line in analysis_text.split("\n"):
                            if "ciudad" in line.lower() or "city" in line.lower():
                                parts = line.split(":")
                                if len(parts) > 1:
                                    address["city"] = parts[1].strip()
                                    break
                    
                    # Buscar coordenadas en cualquier formato
                    coord_pattern = r'(\d+\.\d+)[°\s]*[NS]?,?\s*(\d+\.\d+)[°\s]*[EW]?'
                    coords_match = re.search(coord_pattern, analysis_text)
                    if coords_match:
                        try:
                            coordinates["latitude"] = float(coords_match.group(1))
                            coordinates["longitude"] = float(coords_match.group(2))
                        except (ValueError, IndexError):
                            pass
                    
                    geo_data["merged_data"] = {
                        "coordinates": coordinates,
                        "address": address
                    }
            
            # Si tenemos coordenadas pero no mapa, generar el mapa
            if "merged_data" in geo_data and "coordinates" in geo_data["merged_data"] and "map" not in geo_data:
                coordinates = geo_data["merged_data"]["coordinates"]
                if coordinates["latitude"] is not None and coordinates["longitude"] is not None:
                    try:
                        map_html = geo_service.generate_map(
                            coordinates["latitude"],
                            coordinates["longitude"]
                        )
                        geo_data["map"] = map_html
                    except Exception as map_error:
                        print(f"Error generating map: {str(map_error)}")
        
        # Actualizar el objeto de resultados con los cambios
        results["llm_analysis"] = llm_analysis
        results["geo_data"] = geo_data
        
        # Guardar con indentación para facilitar la lectura
        with open(f"./data/results/{image_id}.json", "w") as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        print(f"Error in background analysis: {str(e)}")
        if image_id in active_sessions:
            active_sessions[image_id]["status"] = "error"
            active_sessions[image_id]["error"] = str(e)

def extract_video_frames_task(video_id: str, file_path: str):
    """Background task to extract frames from a video."""
    try:
        # Update session status
        if video_id in active_sessions:
            active_sessions[video_id]["status"] = "extracting_frames"
        
        # Extract frames
        frames = video_processor.extract_frames(file_path)
        
        # Update session with frame information
        if video_id in active_sessions:
            active_sessions[video_id]["frames"] = frames
            active_sessions[video_id]["frames_count"] = len(frames)
            active_sessions[video_id]["status"] = "frames_extracted"
            
    except Exception as e:
        print(f"Error extracting video frames: {str(e)}")
        if video_id in active_sessions:
            active_sessions[video_id]["status"] = "error"
            active_sessions[video_id]["error"] = str(e) 