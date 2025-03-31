from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
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

# Create FastAPI app
app = FastAPI(
    title="Drone OSINT GeoSpy API",
    description="API for processing drone imagery and video for geolocation analysis"
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

class ImageComparisonRequest(BaseModel):
    image_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

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

# Ensure data directories exist
os.makedirs("./data/uploads", exist_ok=True)
os.makedirs("./data/frames", exist_ok=True)
os.makedirs("./data/results", exist_ok=True)

# Store active analysis sessions
active_sessions = {}

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
async def analyze_image(image_id: str):
    """
    Analyze an uploaded image to extract metadata and perform analysis.
    """
    try:
        # Check if image exists in active sessions
        if image_id not in active_sessions:
            raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")
            
        session = active_sessions[image_id]
        file_path = session["file_path"]
        
        # Update session status
        session["status"] = "analyzing"
        
        # Extract metadata
        metadata = metadata_extractor.extract_metadata(file_path)
        session["metadata"] = metadata
        
        # Extract GPS coordinates from metadata if available
        gps_coords = metadata.get("gps_coordinates")
        
        # Read image file as bytes
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Analyze image with Vision LLM
        llm_analysis = vision_llm.analyze_image(file_path)
        session["llm_analysis"] = llm_analysis
        
        # Process geolocation data
        geo_data = None
        if llm_analysis and "geo_data" in llm_analysis:
            geo_data = llm_analysis["geo_data"]
            
            # Add confidence levels for UI
            confidence = llm_analysis.get("llm_analysis", {}).get("confidence", "low")
            llm_analysis["confidence_level"] = {
                "overall": confidence,
                "country": confidence,
                "city": confidence,
                "district": confidence,
                "neighborhood": confidence,
                "street": confidence,
                "coordinates": confidence
            }
            
            # Add evidence for UI
            llm_analysis["evidence"] = {
                "landmarks": geo_data.get("architectural_features", []),
                "terrain_features": geo_data.get("landscape_features", []),
                "architectural_elements": geo_data.get("architectural_features", []),
                "signage": []
            }
            
            # Add reasoning for UI
            llm_analysis["reasoning"] = llm_analysis.get("llm_analysis", {}).get("description", "No detailed reasoning available.")
            
            # Generate map if coordinates are available
            if "coordinates" in geo_data:
                coords = geo_data["coordinates"]
                lat = coords.get("latitude")
                lon = coords.get("longitude")
                if lat and lon and lat != "0" and lon != "0":
                    try:
                        map_html = geo_service.generate_map(float(lat), float(lon))
                        geo_data["map"] = map_html
                    except Exception as map_error:
                        print(f"Error generating map: {str(map_error)}")
                        geo_data["map"] = None
        
        session["geo_data"] = geo_data
        session["status"] = "completed"
        
        # Crear un objeto con los resultados
        result = AnalysisResponse(
            image_id=image_id,
            metadata=metadata,
            llm_analysis=llm_analysis,
            geo_data=geo_data
        )
        
        # Guardar los resultados en un archivo JSON para facilitar la depuración
        results_file_path = f"./data/results/{image_id}.json"
        with open(results_file_path, "w") as f:
            result_dict = result.dict(exclude_none=True)
            json.dump(result_dict, f, indent=2)
            
        print(f"Saved analysis results to {results_file_path}")
        print(f"LLM analysis has fields: {', '.join(llm_analysis.keys())}")
        if geo_data:
            print(f"Geo data has fields: {', '.join(geo_data.keys())}")
            if "merged_data" in geo_data:
                print(f"Merged data has fields: {', '.join(geo_data['merged_data'].keys())}")
                
        return result
        
    except Exception as e:
        if image_id in active_sessions:
            active_sessions[image_id]["status"] = "error"
            active_sessions[image_id]["error"] = str(e)
        return AnalysisResponse(
            image_id=image_id,
            error=f"Error analyzing image: {str(e)}"
        )

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
    """Generate an interactive map with Mapbox tiles for the given coordinates."""
    try:
        # Generate map using MapboxService
        map_html = mapbox_service.generate_interactive_map(
            latitude=request.latitude,
            longitude=request.longitude,
            zoom=15
        )
        
        if not map_html:
            raise HTTPException(status_code=500, detail="Failed to generate interactive map")
            
        return {"map_html": map_html}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating interactive map: {str(e)}")

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