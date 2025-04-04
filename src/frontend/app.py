import streamlit as st
import requests
import json
import os
import time
import base64
from datetime import datetime
from PIL import Image
import io
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Constants
# Use service name for Docker or fallback to localhost
API_URL = os.environ.get("API_URL", "http://backend:8000")
REFRESH_INTERVAL = 3  # seconds
MAPBOX_TOKEN = os.environ.get("MAPBOX_API_KEY", "pk.eyJ1IjoiaW5mYW50ZXNyb21lcm9hZHJpYW4iLCJhIjoiY2xncXAwYWJzMGJlNDNmb2h2MTBvNDF1YiJ9.nXs806Ka9zFqtbKQ4an41Q")

# Initialize session state
def initialize_session_state():
    """Initialize the session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_image_id' not in st.session_state:
        st.session_state.current_image_id = None
    if 'current_stream_id' not in st.session_state:
        st.session_state.current_stream_id = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "upload"
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'map_html' not in st.session_state:
        st.session_state.map_html = None
    if 'is_streaming' not in st.session_state:
        st.session_state.is_streaming = False
    if 'map_style' not in st.session_state:
        st.session_state.map_style = "satellite-streets-v11"

# Initialize session state at the beginning
initialize_session_state()

# Set Streamlit theme to dark mode
st.set_page_config(
    page_title="DRONE OSINT GEOSPY",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        # Drone OSINT GeoSpy
        
        Version 1.0.0  
        Security Level: CLASSIFIED
        """
    }
)

# Add Mapbox GL JS resources
st.markdown(f"""
<head>
    <link href="https://api.mapbox.com/mapbox-gl-js/v2.14.1/mapbox-gl.css" rel="stylesheet">
    <script src="https://api.mapbox.com/mapbox-gl-js/v2.14.1/mapbox-gl.js"></script>
    <style>
        .mapboxgl-map {{
            border-radius: 5px;
            height: 500px;
        }}
    </style>
</head>
""", unsafe_allow_html=True)

# Apply military style
def apply_military_style():
    """Apply military-style CSS to the Streamlit app."""
    st.markdown("""
    <style>
        /* Main colors */
        :root {
            --military-dark: #0F1C2E;
            --military-green: #1E3F20;
            --military-light: #8B9386;
            --accent-red: #A52A2A;
            --accent-yellow: #DAA520;
            --text-color: #E0E0E0;
        }
        
        /* Base styling */
        .stApp {
            background-color: var(--military-dark);
            color: var(--text-color);
        }
        
        /* Headers */
        h1, h2, h3 {
            color: var(--text-color) !important;
            font-family: 'Courier New', monospace !important;
            text-transform: uppercase;
        }
        
        h1 {
            border-bottom: 2px solid var(--accent-red);
            padding-bottom: 10px;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: var(--military-green);
        }
        
        /* Buttons */
        .stButton>button {
            background-color: var(--military-green);
            color: var(--text-color);
            border: 1px solid var(--accent-yellow);
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        .stButton>button:hover {
            background-color: var(--accent-red);
            border: 1px solid var(--text-color);
        }
        
        /* Info boxes */
        .info-box {
            background-color: rgba(30, 63, 32, 0.7);
            border: 1px solid var(--accent-yellow);
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        .status-green {
            color: #4CAF50;
        }
        
        .status-red {
            color: #F44336;
        }
        
        .status-yellow {
            color: #FFC107;
        }
        
        /* Timestamp */
        .timestamp {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: var(--accent-yellow);
            text-align: right;
        }
        
        /* Input fields */
        .stTextInput>div>div>input {
            background-color: #1E3F20;
            color: var(--text-color);
            border: 1px solid var(--accent-yellow);
        }
        
        /* Maps container */
        .map-container {
            border: 2px solid var(--accent-yellow);
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
            background-color: rgba(30, 63, 32, 0.7);
        }
        
        .map-container h3 {
            margin-top: 0;
            margin-bottom: 10px;
            color: var(--accent-yellow) !important;
        }
        
        .map-container iframe {
            width: 100%;
            height: 400px;
            border: none;
            border-radius: 5px;
            background-color: white;
        }
        
        /* Chat messages */
        .chat-container {
            border: 1px solid var(--accent-yellow);
            border-radius: 5px;
            padding: 10px;
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        
        .user-message {
            background-color: var(--military-green);
            color: var(--text-color);
            padding: 12px;
            border-radius: 5px;
            margin-bottom: 12px;
            text-align: right;
        }
        
        .assistant-message {
            background-color: rgba(30, 63, 32, 0.7);
            color: var(--text-color);
            padding: 12px;
            border-radius: 5px;
            margin-bottom: 12px;
            border-left: 3px solid var(--accent-yellow);
            line-height: 1.6;
            white-space: pre-wrap;
        }
        
        .assistant-message strong {
            color: var(--accent-yellow);
            font-weight: bold;
        }
        
        .assistant-message br {
            display: block;
            content: "";
            margin-top: 10px;
        }
        
        /* Coordinates display */
        .coordinates {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            color: var(--accent-yellow);
            padding: 5px;
            background-color: rgba(15, 28, 46, 0.7);
            border-radius: 3px;
        }
    </style>
    """, unsafe_allow_html=True)

apply_military_style()

# Header with blinking status
def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <h1>DRONE OSINT GEOSPY</h1>
    <div class="timestamp">
        MISSION TIME: {get_current_time()} | 
        <span class="status-green">●</span> SYSTEM OPERATIONAL
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## MISSION CONTROL")
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown(f"**SYSTEM STATUS**: <span class='status-green'>OPERATIONAL</span>", unsafe_allow_html=True)
    st.markdown(f"**SESSION ID**: {str(hash(datetime.now()))[:8]}", unsafe_allow_html=True)
    
    if st.session_state.current_image_id:
        st.markdown(f"**ACTIVE IMAGE**: {st.session_state.current_image_id[:8]}...", unsafe_allow_html=True)
    
    if st.session_state.current_stream_id:
        stream_status = "ACTIVE" if st.session_state.is_streaming else "DISCONNECTED"
        status_class = "status-green" if st.session_state.is_streaming else "status-red"
        st.markdown(f"**STREAM STATUS**: <span class='{status_class}'>{stream_status}</span>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("## SYSTEM OPTIONS")
    
    # Navigation tabs
    if st.button("👁️ IMAGE ANALYSIS", use_container_width=True):
        st.session_state.active_tab = "upload"
    
    if st.button("🎥 DRONE STREAM", use_container_width=True):
        st.session_state.active_tab = "stream"
    
    if st.button("🔍 INTERROGATION", use_container_width=True):
        st.session_state.active_tab = "chat"
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Additional system info
    st.markdown("## SYSTEM DETAILS")
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("**VISION MODEL**: Gemini 2.5 Pro (exp-03-25)")
    st.markdown("**OBJECT DETECTION**: YOLOv8 (Neural Network)")
    st.markdown("**ACCURACY**: HIGH")
    st.markdown("**RESPONSE TIME**: 1-3s")
    st.markdown("**FEATURES**: <span style='color: #1a73e8;'>Advanced Geospatial Analysis</span> | <span style='color: #FF4B4B;'>Object & Personnel Detection</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Main content area
if st.session_state.active_tab == "upload":
    st.markdown("## 👁️ TERRAIN INTELLIGENCE ANALYSIS")
    
    # Add location search
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("### 🔍 SEARCH LOCATION")
    
    location_search = st.text_input("Enter location (city, address, landmark)", key="location_search")
    
    if location_search:
        with st.spinner("Searching location..."):
            # Call geocoding API
            geocode_response = requests.post(
                f"{API_URL}/api/geocode/forward",
                json={"query": location_search, "limit": 5}
            )
            
            if geocode_response.status_code == 200:
                results = geocode_response.json().get("results", [])
                
                if results:
                    st.markdown("**SEARCH RESULTS:**")
                    
                    # Create a list of locations for the dropdown
                    location_options = [f"{r['place_name']} [{r['longitude']:.6f}, {r['latitude']:.6f}]" for r in results]
                    selected_location = st.selectbox("Select a location", location_options)
                    
                    if selected_location:
                        # Extract the selected location index
                        selected_idx = location_options.index(selected_location)
                        selected_result = results[selected_idx]
                        
                        # Display selected location details
                        st.markdown(f"**SELECTED**: {selected_result['place_name']}")
                        st.markdown(f"**COORDINATES**: {selected_result['latitude']:.6f}, {selected_result['longitude']:.6f}")
                        
                        # Display the map for the selected location
                        if st.button("📍 SHOW ON MAP", key="show_location_map"):
                            # Generate map with Mapbox GL JS
                            map_html = create_mapbox_map(
                                selected_result['latitude'], 
                                selected_result['longitude'],
                                zoom=14
                            )
                            st.markdown("<div class='map-container'>", unsafe_allow_html=True)
                            st.markdown("<h3>LOCATION MAP</h3>", unsafe_allow_html=True)
                            st.components.v1.html(map_html, height=600, scrolling=False)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Get static satellite image
                            static_response = requests.post(
                                f"{API_URL}/api/static-map",
                                json={
                                    "latitude": selected_result['latitude'],
                                    "longitude": selected_result['longitude']
                                },
                                params={"style": "satellite-v9", "zoom": 14}
                            )
                            
                            if static_response.status_code == 200:
                                static_data = static_response.json()
                                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                                st.markdown("### SATELLITE VIEW")
                                st.markdown(f"<img src='{static_data['image_data']}' style='width:100%;border-radius:5px;'>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No results found for your search.")
            else:
                st.error(f"Error searching location: {geocode_response.text}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("UPLOAD RECONNAISSANCE IMAGE", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("### IMAGE DETAILS")
            st.markdown(f"**FILENAME**: {uploaded_file.name}")
            st.markdown(f"**SIZE**: {uploaded_file.size / 1024:.1f} KB")
            st.markdown(f"**DIMENSIONS**: {image.width} x {image.height}")
            st.markdown(f"**UPLOAD TIME**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Analysis button
            if st.button("🎯 ANALYZE TERRAIN", use_container_width=True):
                # Create a placeholder for the progress bar
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Initialize progress bar
                progress_bar = progress_placeholder.progress(0)
                status_placeholder.info("Iniciando análisis...")
                
                try:
                    # Save the image to a BytesIO object (5%)
                    progress_bar.progress(5)
                    status_placeholder.info("Preparando imagen...")
                    
                    # Try to determine the original format or fallback to PNG
                    if hasattr(image, 'format') and image.format:
                        img_format = image.format
                    else:
                        # Convert to RGB mode if not already to avoid transparency issues
                        if image.mode == 'RGBA':
                            image = image.convert('RGB')
                        img_format = 'PNG'  # PNG is more universally supported
                        
                    buf = io.BytesIO()
                    try:
                        image.save(buf, format=img_format)
                    except Exception as save_error:
                        # If the original format fails, try PNG as fallback
                        print(f"Error saving with original format: {str(save_error)}")
                        if img_format != 'PNG':
                            if image.mode == 'RGBA':
                                image = image.convert('RGB')
                            buf = io.BytesIO()  # Create a new buffer
                            image.save(buf, format='PNG')
                        else:
                            # If PNG also fails, raise the error
                            raise save_error
                            
                    byte_im = buf.getvalue()
                    
                    # Upload to the API (15%)
                    progress_bar.progress(15)
                    status_placeholder.info("Subiendo imagen al servidor...")
                    
                    content_type = f"image/{img_format.lower()}" if img_format != 'PNG' else "image/png"
                    files = {"file": (uploaded_file.name, byte_im, content_type)}
                    response = requests.post(f"{API_URL}/api/upload/image", files=files)
                    
                    if response.status_code == 200:
                        upload_data = response.json()
                        image_id = upload_data.get("image_id")
                        st.session_state.current_image_id = image_id
                        
                        # Analyze the image (30%)
                        progress_bar.progress(30)
                        status_placeholder.info("Analizando imagen con inteligencia artificial...")
                        
                        analysis_response = requests.post(f"{API_URL}/api/analyze/image/{image_id}")
                        
                        if analysis_response.status_code == 200:
                            # Processing analysis data (60%)
                            progress_bar.progress(60)
                            status_placeholder.info("Procesando resultados del análisis...")
                            
                            analysis_data = analysis_response.json()
                            
                            # Extract data with error handling
                            try:
                                # LLM Analysis
                                llm_analysis = analysis_data.get('llm_analysis', {})
                                if llm_analysis is None:
                                    llm_analysis = {}
                                
                                if isinstance(llm_analysis, str):
                                    description = llm_analysis
                                    confidence = 'medium'  # Default confidence for string responses
                                else:
                                    description = llm_analysis.get('description', 'No description available')
                                    confidence = llm_analysis.get('confidence', 'low')
                                
                                # Geo Data
                                geo_data = analysis_data.get('geo_data', {})
                                if geo_data is None:
                                    geo_data = {}
                                    
                                if isinstance(geo_data, str):
                                    # If geo_data is a string, try to parse it as JSON
                                    try:
                                        geo_data = json.loads(geo_data)
                                    except json.JSONDecodeError:
                                        geo_data = {'text_analysis': geo_data}
                                
                                # Extract location data with safe defaults
                                country = geo_data.get('country', 'Unknown')
                                city = geo_data.get('city', 'Unknown')
                                neighborhood = geo_data.get('neighborhood', 'Unknown')
                                street = geo_data.get('street', 'Unknown')
                                coordinates = geo_data.get('coordinates', {})
                                if coordinates is None:
                                    coordinates = {}
                                
                                lat = coordinates.get('latitude', '0')
                                lon = coordinates.get('longitude', '0')
                                architectural_features = geo_data.get('architectural_features', [])
                                if architectural_features is None:
                                    architectural_features = []
                                
                                landscape_features = geo_data.get('landscape_features', [])
                                if landscape_features is None:
                                    landscape_features = []
                                
                                # Metadata with safe defaults
                                metadata = analysis_data.get('metadata', {})
                                if metadata is None:
                                    metadata = {}
                                
                                camera_info = metadata.get('camera_info', {})
                                if camera_info is None:
                                    camera_info = {}
                                
                                make = camera_info.get('make', 'Unknown')
                                model = camera_info.get('model', 'Unknown')
                                focal_length = camera_info.get('focal_length', 'Unknown')
                                exposure_time = camera_info.get('exposure_time', 'Unknown')
                                f_number = camera_info.get('f_number', 'Unknown')
                                iso = camera_info.get('iso', 'Unknown')
                                
                                gps_info = metadata.get('gps_info', {})
                                if gps_info is None:
                                    gps_info = {}
                                
                                gps_lat = gps_info.get('latitude', 'Unknown')
                                gps_lon = gps_info.get('longitude', 'Unknown')
                                gps_alt = gps_info.get('altitude', 'Unknown')
                                
                                # Update session state with analysis results
                                st.session_state.analysis_results = {
                                    'description': description,
                                    'confidence': confidence,
                                    'location': {
                                        'country': country,
                                        'city': city,
                                        'neighborhood': neighborhood,
                                        'street': street,
                                        'coordinates': {
                                            'latitude': lat,
                                            'longitude': lon
                                        }
                                    },
                                    'features': {
                                        'architectural': architectural_features,
                                        'landscape': landscape_features
                                    },
                                    'metadata': {
                                        'camera': {
                                            'make': make,
                                            'model': model,
                                            'focal_length': focal_length,
                                            'exposure_time': exposure_time,
                                            'f_number': f_number,
                                            'iso': iso
                                        },
                                        'gps': {
                                            'latitude': gps_lat,
                                            'longitude': gps_lon,
                                            'altitude': gps_alt
                                        }
                                    }
                                }
                                
                                # Generate map if not already present (85%)
                                progress_bar.progress(85)
                                status_placeholder.info("Generando mapa interactivo...")
                                
                                if not st.session_state.map_html and lat != '0' and lon != '0':
                                    try:
                                        # Call the API to generate the map
                                        response = requests.post(
                                            f"{API_URL}/api/generate/interactive_map",
                                            json={"latitude": float(lat), "longitude": float(lon), "zoom": 13}
                                        )
                                        
                                        if response.status_code == 200:
                                            map_data = response.json()
                                            st.session_state.map_html = map_data.get("map_html")
                                    except Exception as api_err:
                                        print(f"Fallback map generation also failed: {str(api_err)}")
                                        status_placeholder.error(f"No se pudo generar el mapa: {str(api_err)}")
                                
                                # Save results to JSON file (95%)
                                progress_bar.progress(95)
                                status_placeholder.info("Guardando resultados...")
                                
                                results_file = os.path.join('data', 'results', f'{image_id}.json')
                                with open(results_file, 'w', encoding='utf-8') as f:
                                    json.dump(analysis_data, f, ensure_ascii=False, indent=2)
                                
                                # Analysis complete (100%)
                                progress_bar.progress(100)
                                status_placeholder.success("¡Análisis completado con éxito!")
                                
                                # Clear the progress bar after a delay
                                time.sleep(1)
                                progress_placeholder.empty()
                                status_placeholder.empty()
                                
                                # Display success message
                                st.success("Análisis completado con éxito!")
                                
                                # Refresh the page to display results
                                st.rerun()
                                
                            except Exception as e:
                                progress_bar.progress(100)
                                status_placeholder.error(f"Error procesando resultados: {str(e)}")
                                st.error(f"Error procesando resultados del análisis: {str(e)}")
                                print(f"Error details: {str(e)}")
                                print(f"Analysis data structure: {json.dumps(analysis_data, indent=2)}")
                        else:
                            progress_bar.progress(100)
                            status_placeholder.error(f"ERROR ANALIZANDO IMAGEN: {analysis_response.text}")
                            st.error(f"ERROR ANALIZANDO IMAGEN: {analysis_response.text}")
                    else:
                        progress_bar.progress(100)
                        status_placeholder.error(f"ERROR SUBIENDO IMAGEN: {response.text}")
                        st.error(f"ERROR SUBIENDO IMAGEN: {response.text}")
                except Exception as e:
                    if 'progress_bar' in locals():
                        progress_bar.progress(100)
                    if 'status_placeholder' in locals():
                        status_placeholder.error(f"Error procesando imagen: {str(e)}")
                    st.error(f"Error procesando imagen: {str(e)}")
                    print(f"Image processing error details: {str(e)}")
    
    # Display analysis results
    if st.session_state.analysis_results:
        st.markdown("## 📊 INTELLIGENCE REPORT")
        
        # Mostrar el análisis en bruto en caso de que la visualización falle
        with st.expander("Raw Analysis Data (Debug)", expanded=False):
            st.json(st.session_state.analysis_results)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Map
            if st.session_state.analysis_results:
                location = st.session_state.analysis_results.get('location', {})
                if location is None:
                    location = {}
                
                coordinates = location.get('coordinates', {})
                if coordinates is None:
                    coordinates = {}
                
                lat = coordinates.get('latitude')
                lon = coordinates.get('longitude')
                
                if lat and lon and lat != 0 and lon != 0:
                    try:
                        # Call the API to generate the map
                        response = requests.post(
                            f"{API_URL}/api/generate/interactive_map",
                            json={"latitude": float(lat), "longitude": float(lon), "zoom": 13}
                        )
                        
                        if response.status_code == 200:
                            map_data = response.json()
                            map_html = map_data.get("map_html")
                            
                            if map_html:
                                st.markdown("<div class='map-container'>", unsafe_allow_html=True)
                                st.markdown("<h3>GEOLOCATION DATA</h3>", unsafe_allow_html=True)
                                st.components.v1.html(map_html, height=600, scrolling=False)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Add comparison button
                                if st.button("🔍 COMPARE WITH SATELLITE/MAPS", key="compare_button", use_container_width=True):
                                    with st.spinner("Generating comparison..."):
                                        # Get current image ID
                                        image_id = st.session_state.current_image_id
                                        
                                        # Call comparison API
                                        comparison_response = requests.post(
                                            f"{API_URL}/api/location/compare",
                                            json={
                                                "image_id": image_id,
                                                "latitude": float(lat),
                                                "longitude": float(lon)
                                            }
                                        )
                                        
                                        if comparison_response.status_code == 200:
                                            comparison_data = comparison_response.json()
                                            comparison_html = comparison_data.get("comparison_html")
                                            
                                            if comparison_html:
                                                st.markdown("<div class='comparison-section'>", unsafe_allow_html=True)
                                                st.markdown("<h3>IMAGE COMPARISON</h3>", unsafe_allow_html=True)
                                                st.components.v1.html(comparison_html, height=800, scrolling=True)
                                                st.markdown("</div>", unsafe_allow_html=True)
                                            else:
                                                st.warning("Failed to generate comparison")
                                        else:
                                            st.error(f"Error generating comparison: {comparison_response.text}")
                            else:
                                st.warning("No map data available")
                        else:
                            st.error(f"Error generating map: {response.text}")
                    except Exception as e:
                        st.error(f"Error displaying map: {str(e)}")
            
            # Location details
            location = st.session_state.analysis_results.get('location', {})
            if location and isinstance(location, dict):
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown("### LOCATION DETAILS")
                st.markdown(f"**COUNTRY**: {location.get('country', 'Unknown')}")
                st.markdown(f"**CITY**: {location.get('city', 'Unknown')}")
                st.markdown(f"**NEIGHBORHOOD**: {location.get('neighborhood', 'Unknown')}")
                st.markdown(f"**STREET**: {location.get('street', 'Unknown')}")
                
                coords = location.get('coordinates', {})
                if coords and isinstance(coords, dict):
                    try:
                        lat = float(coords.get('latitude', 0))
                        lon = float(coords.get('longitude', 0))
                        # Format coordinates with 6 decimal places for precision
                        st.markdown(f"**COORDINATES**: {lat:.6f}, {lon:.6f}")
                        
                        # Generate map if not already present
                        if not st.session_state.map_html and lat != 0 and lon != 0:
                            try:
                                # Generate map with Mapbox GL JS
                                st.session_state.map_html = create_mapbox_map(lat, lon)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating map: {str(e)}")
                    except (ValueError, TypeError) as e:
                        st.warning(f"Invalid coordinates format: {coords}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Description and confidence
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("### ANALYSIS SUMMARY")
            st.markdown(f"**DESCRIPTION**: {st.session_state.analysis_results.get('description', 'No description available')}")
            st.markdown(f"**CONFIDENCE**: {st.session_state.analysis_results.get('confidence', 'low')}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Features
            features = st.session_state.analysis_results.get('features', {})
            if features and isinstance(features, dict):
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown("### TERRAIN FEATURES")
                
                architectural = features.get('architectural')
                if architectural and isinstance(architectural, list):
                    st.markdown("**ARCHITECTURAL FEATURES:**")
                    for feature in architectural:
                        st.markdown(f"- {feature}")
                
                landscape = features.get('landscape')
                if landscape and isinstance(landscape, list):
                    st.markdown("**LANDSCAPE FEATURES:**")
                    for feature in landscape:
                        st.markdown(f"- {feature}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Metadata
            metadata = st.session_state.analysis_results.get('metadata', {})
            if metadata and isinstance(metadata, dict):
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown("### TECHNICAL DATA")
                
                camera = metadata.get('camera', {})
                if camera and isinstance(camera, dict):
                    st.markdown("**CAMERA INFO:**")
                    st.markdown(f"- Make: {camera.get('make', 'Unknown')}")
                    st.markdown(f"- Model: {camera.get('model', 'Unknown')}")
                    st.markdown(f"- Focal Length: {camera.get('focal_length', 'Unknown')}")
                    st.markdown(f"- Exposure: {camera.get('exposure_time', 'Unknown')}")
                    st.markdown(f"- F-Number: {camera.get('f_number', 'Unknown')}")
                    st.markdown(f"- ISO: {camera.get('iso', 'Unknown')}")
                
                gps = metadata.get('gps', {})
                if gps and isinstance(gps, dict):
                    st.markdown("**GPS DATA:**")
                    st.markdown(f"- Latitude: {gps.get('latitude', 'Unknown')}")
                    st.markdown(f"- Longitude: {gps.get('longitude', 'Unknown')}")
                    st.markdown(f"- Altitude: {gps.get('altitude', 'Unknown')}")
                
                st.markdown("</div>", unsafe_allow_html=True)

    # Add new section to show object detection results after analyzing the image
    if st.session_state.analysis_results:
        # Object detection button
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Add a button to trigger object detection
            if st.button("🔍 DETECT OBJECTS & PERSONNEL", key="detect_objects_btn", use_container_width=True):
                with st.spinner("PROCESSING OBJECT DETECTION..."):
                    # Get current image ID
                    image_id = st.session_state.current_image_id
                    
                    # Call object detection API
                    detection_response = requests.post(
                        f"{API_URL}/api/detect/objects",
                        json={
                            "image_id": image_id,
                            "confidence_threshold": 0.25
                        }
                    )
                    
                    if detection_response.status_code == 200:
                        detection_data = detection_response.json()
                        
                        # Store detection results in session state
                        st.session_state.object_detection_results = detection_data.get("detection_results", {})
                        
                        # Get annotated image immediately
                        annotated_image_response = requests.get(f"{API_URL}/api/image/annotated/{image_id}")
                        
                        if annotated_image_response.status_code == 200:
                            annotated_image_data = annotated_image_response.json()
                            st.session_state.annotated_image = annotated_image_data.get("annotated_image")
                            st.success(f"Object detection completed: {detection_data.get('message')}")
                        else:
                            st.warning(f"Detection successful but couldn't load annotated image: {annotated_image_response.text}")
                            st.success(f"Object detection completed: {detection_data.get('message')}")
                    else:
                        st.error(f"Error in object detection: {detection_response.text}")

        # Display object detection results if available
        if 'object_detection_results' in st.session_state and st.session_state.object_detection_results:
            st.markdown("## 📊 OBJECT DETECTION RESULTS")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Display annotated image if available
                if 'annotated_image' in st.session_state and st.session_state.annotated_image:
                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                    st.markdown("### ANNOTATED IMAGE")
                    st.markdown(f"<img src='{st.session_state.annotated_image}' style='width:100%; border-radius:5px;'>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # Try to get the annotated image path directly from the results
                    try:
                        annotated_img_path = st.session_state.object_detection_results.get("annotated_image_path")
                        if annotated_img_path:
                            st.warning("Trying to load annotated image directly...")
                            image_id = st.session_state.current_image_id
                            # Make a direct request to get the annotated image
                            annotated_image_response = requests.get(f"{API_URL}/api/image/annotated/{image_id}")
                            if annotated_image_response.status_code == 200:
                                annotated_image_data = annotated_image_response.json()
                                st.session_state.annotated_image = annotated_image_data.get("annotated_image")
                                # Display the image
                                if st.session_state.annotated_image:
                                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                                    st.markdown("### ANNOTATED IMAGE")
                                    st.markdown(f"<img src='{st.session_state.annotated_image}' style='width:100%; border-radius:5px;'>", unsafe_allow_html=True)
                                    st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.error(f"Error obteniendo imagen anotada: {annotated_image_response.text}")
                        else:
                            st.error("Error: No se encontró la ruta de la imagen anotada.")
                    except Exception as e:
                        st.error(f"Error al procesar la imagen anotada: {str(e)}")
            
            with col2:
                # Display detection summary
                summary = st.session_state.object_detection_results.get("summary", {})
                if summary:
                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                    st.markdown("### DETECTION SUMMARY")
                    st.markdown(f"**TOTAL OBJECTS**: {summary.get('total_objects_detected', 0)}")
                    st.markdown(f"**PERSONNEL PRESENT**: {'Yes' if summary.get('has_people', False) else 'No'}")
                    
                    if summary.get('most_common_object'):
                        st.markdown(f"**MOST COMMON**: {summary.get('most_common_object')}")
                    
                    # Show object counts
                    object_counts = summary.get('object_counts', {})
                    if object_counts:
                        st.markdown("### OBJECT COUNTS")
                        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
                            st.markdown(f"- {obj}: {count}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display tactical assessment
                assessment = st.session_state.object_detection_results.get("assessment", {})
                if assessment:
                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                    st.markdown("### TACTICAL ASSESSMENT")
                    
                    # Format threat level with appropriate color
                    threat_level = assessment.get('threat_level', 'UNKNOWN')
                    threat_color = "#4CAF50" if threat_level == "LOW" else "#FFC107" if threat_level == "MEDIUM" else "#F44336" if threat_level == "HIGH" else "#9E9E9E"
                    st.markdown(f"**THREAT LEVEL**: <span style='color:{threat_color};font-weight:bold;'>{threat_level}</span>", unsafe_allow_html=True)
                    
                    # Format other assessment data
                    st.markdown(f"**TACTICAL VALUE**: {assessment.get('tactical_value', 'UNKNOWN')}")
                    st.markdown(f"**INTELLIGENCE VALUE**: {assessment.get('intelligence_value', 'UNKNOWN')}")
                    st.markdown(f"**POPULATION DENSITY**: {assessment.get('population_density', 'UNKNOWN')}")
                    st.markdown(f"**VEHICLE PRESENCE**: {assessment.get('vehicle_presence', 'UNKNOWN')}")
                    st.markdown(f"**INFRASTRUCTURE**: {assessment.get('infrastructure_assessment', 'UNKNOWN')}")
                    st.markdown(f"**PERSONNEL COUNT**: {assessment.get('personnel_count', 0)}")
                    st.markdown(f"**VEHICLE COUNT**: {assessment.get('vehicle_count', 0)}")
                    
                    # Show warning if suspicious activities detected
                    if assessment.get('suspicious_activities', False):
                        st.markdown(f"<div style='background-color:#F44336;color:white;padding:10px;border-radius:5px;margin-top:10px;'><strong>WARNING:</strong> Potential suspicious activities detected</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.active_tab == "stream":
    st.markdown("## 🎥 LIVE DRONE RECONNAISSANCE")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Stream connection
        if not st.session_state.is_streaming:
            stream_url = st.text_input("ENTER DRONE STREAM URL (RTSP/HTTP)")
            
            if st.button("🔄 CONNECT TO DRONE FEED", use_container_width=True):
                if stream_url:
                    with st.spinner("ESTABLISHING CONNECTION..."):
                        # Connect to stream
                        data = {"stream_url": stream_url}
                        response = requests.post(f"{API_URL}/api/stream/connect", data=data)
                        
                        if response.status_code == 200:
                            stream_data = response.json()
                            st.session_state.current_stream_id = stream_data.get("stream_id")
                            st.session_state.is_streaming = True
                            st.success("DRONE FEED CONNECTED")
                        else:
                            st.error(f"CONNECTION ERROR: {response.text}")
                else:
                    st.warning("PLEASE ENTER A VALID STREAM URL")
        else:
            # Display stream
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("### LIVE FEED STATUS")
            st.markdown(f"**STREAM ID**: {st.session_state.current_stream_id[:8]}...")
            st.markdown(f"**STATUS**: <span class='status-green'>ACTIVE</span>", unsafe_allow_html=True)
            st.markdown(f"**START TIME**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Placeholder for the live stream view
            stream_placeholder = st.empty()
            
            # Disconnect button
            if st.button("⏹️ DISCONNECT", use_container_width=True):
                if st.session_state.current_stream_id:
                    with st.spinner("TERMINATING CONNECTION..."):
                        response = requests.post(f"{API_URL}/api/stream/disconnect/{st.session_state.current_stream_id}")
                        
                        if response.status_code == 200:
                            st.session_state.is_streaming = False
                            st.session_state.current_stream_id = None
                            st.success("DRONE FEED DISCONNECTED")
                        else:
                            st.error(f"DISCONNECTION ERROR: {response.text}")
            
            # Analyze frame button
            if st.button("🔍 CAPTURE & ANALYZE FRAME", use_container_width=True):
                if st.session_state.current_stream_id:
                    with st.spinner("CAPTURING AND ANALYZING FRAME..."):
                        response = requests.get(
                            f"{API_URL}/api/stream/latest-frame/{st.session_state.current_stream_id}",
                            params={"analyze": True}
                        )
                        
                        if response.status_code == 200:
                            frame_data = response.json()
                            
                            if frame_data.get("status") == "analyzed":
                                st.session_state.analysis_results = {
                                    "metadata": frame_data.get("metadata"),
                                    "llm_analysis": frame_data.get("llm_analysis"),
                                    "geo_data": frame_data.get("geo_data")
                                }
                                
                                # Extract map HTML if available
                                if "geo_data" in frame_data and "map" in frame_data["geo_data"]:
                                    st.session_state.map_html = frame_data["geo_data"]["map"]
                                
                                st.success("FRAME ANALYSIS COMPLETE")
                            else:
                                st.warning(f"FRAME STATUS: {frame_data.get('status')}")
                        else:
                            st.error(f"FRAME CAPTURE ERROR: {response.text}")
            
            # Update the stream view periodically
            while st.session_state.is_streaming:
                try:
                    response = requests.get(f"{API_URL}/api/stream/latest-frame/{st.session_state.current_stream_id}")
                    
                    if response.status_code == 200:
                        frame_data = response.json()
                        
                        if frame_data.get("status") != "no_frame":
                            # Display the frame
                            file_path = frame_data.get("file_path")
                            if file_path:
                                # For local development, we'd need to convert the path to a web URL
                                # In production, we'd use a proper file serving mechanism
                                stream_placeholder.markdown(f"<img src='/data/frames/{os.path.basename(file_path)}' width='100%'>", unsafe_allow_html=True)
                    
                    # Sleep to avoid overloading
                    time.sleep(REFRESH_INTERVAL)
                except Exception as e:
                    st.error(f"ERROR UPDATING STREAM: {str(e)}")
                    break
    
    with col2:
        if st.session_state.analysis_results and st.session_state.is_streaming:
            # Display analysis similar to the upload tab
            if "geo_data" in st.session_state.analysis_results and st.session_state.analysis_results["geo_data"]:
                geo_data = st.session_state.analysis_results["geo_data"]
                
                if "merged_data" in geo_data and "coordinates" in geo_data["merged_data"]:
                    coords = geo_data["merged_data"]["coordinates"]
                    lat_value = coords.get('latitude')
                    long_value = coords.get('longitude')
                    
                    # Manejar valores None o no válidos
                    lat_display = f"{lat_value:.6f}" if lat_value is not None else "N/A"
                    long_display = f"{long_value:.6f}" if long_value is not None else "N/A"
                    
                    st.markdown(f"""
                    <div class='coordinates'>
                        LAT: {lat_display} | 
                        LONG: {long_display}
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.session_state.map_html:
                    st.markdown("<div class='map-container'>", unsafe_allow_html=True)
                    st.markdown("<h3>GEOLOCATION DATA</h3>", unsafe_allow_html=True)
                    
                    # Create a unique ID for this map instance
                    map_id = f"map_{hash(datetime.now())}"
                    
                    # Add the map HTML with proper structure
                    st.markdown(f"""
                        <div style="width:100%; height:400px; position:relative;">
                            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css"/>
                            <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
                            <div id="{map_id}" style="width:100%; height:100%; position:relative;"></div>
                            <script>
                                var map = L.map('{map_id}').setView([-13.5167, -71.9788], 15);
                                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                                    attribution: '© OpenStreetMap contributors'
                                }}).addTo(map);
                                
                                var marker = L.marker([-13.5167, -71.9788]).addTo(map);
                                marker.bindPopup("Plaza de Armas de Cusco").openPopup();
                                
                                var circle = L.circle([-13.5167, -71.9788], {{
                                    color: 'red',
                                    fillColor: '#f03',
                                    fillOpacity: 0.2,
                                    radius: 50
                                }}).addTo(map);
                            </script>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                if "merged_data" in geo_data and "address" in geo_data["merged_data"]:
                    address = geo_data["merged_data"]["address"]
                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                    st.markdown("### LOCATION DETAILS")
                    st.markdown(f"**COUNTRY**: {address.get('country', 'Unknown')}")
                    st.markdown(f"**CITY**: {address.get('city', 'Unknown')}")
                    st.markdown(f"**DISTRICT**: {address.get('district', 'Unknown')}")
                    st.markdown(f"**NEIGHBORHOOD**: {address.get('neighborhood', 'Unknown')}")
                    st.markdown(f"**STREET**: {address.get('street', 'Unknown')}")
                    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.active_tab == "chat":
    st.markdown("## 🔍 TERRAIN INTERROGATION")
    
    if not st.session_state.current_image_id and not st.session_state.is_streaming:
        st.warning("UPLOAD AN IMAGE OR CONNECT TO A STREAM FIRST")
    else:
        # Determine what we're chatting about
        chat_target = None
        target_name = ""
        
        if st.session_state.current_image_id:
            chat_target = st.session_state.current_image_id
            target_name = "IMAGE"
        elif st.session_state.current_stream_id:
            # Use the latest frame from the stream
            response = requests.get(f"{API_URL}/api/stream/latest-frame/{st.session_state.current_stream_id}")
            if response.status_code == 200:
                frame_data = response.json()
                if frame_data.get("status") != "no_frame":
                    chat_target = frame_data.get("frame_id")
                    target_name = "STREAM FRAME"
        
        if chat_target:
            st.markdown(f"### INTERROGATING {target_name}")
            
            # Display chat history
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
                else:
                    # Format the response with markdown and styling
                    response_content = message['content']
                    # Ensure we're working with a string
                    if isinstance(response_content, dict):
                        response_content = response_content.get('response', str(response_content))
                    # Convert the response to string and process markdown
                    response_text = str(response_content)
                    # Use markdown to handle the formatting
                    st.markdown(f"<div class='assistant-message'>{response_text}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Input field
            user_message = st.text_input("ENTER QUERY ABOUT TERRAIN:", key="chat_input")
            
            if st.button("🔍 SEND QUERY", use_container_width=True):
                if user_message:
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_message})
                    
                    with st.spinner("PROCESSING QUERY..."):
                        # Send to API
                        data = {
                            "image_id": chat_target,
                            "message": user_message
                        }
                        
                        response = requests.post(f"{API_URL}/api/chat/image/{chat_target}", json=data)
                        
                        if response.status_code == 200:
                            chat_data = response.json()
                            
                            # Get the response text
                            response_text = chat_data.get("response", "ERROR: No response received")
                            if isinstance(response_text, dict):
                                response_text = response_text.get('response', str(response_text))
                            
                            # Create a placeholder for the streaming response
                            response_placeholder = st.empty()
                            
                            # Stream the response character by character
                            full_response = ""
                            for char in str(response_text):
                                full_response += char
                                response_placeholder.markdown(f"<div class='assistant-message'>{full_response}</div>", unsafe_allow_html=True)
                                time.sleep(0.01)  # Ajustado para ser más rápido
                            
                            # Add the complete response to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                            
                            # Clear the input field (need to rerun the app)
                            st.rerun()
                        else:
                            st.error(f"ERROR: {response.text}")
        else:
            st.warning("NO VALID IMAGE OR FRAME FOUND")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: var(--accent-yellow); font-size: 12px;">
    DRONE OSINT GEOSPY v1.0 | CLASSIFIED | FOR AUTHORIZED PERSONNEL ONLY
</div>
""", unsafe_allow_html=True)

# Add custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
        /* General Styles */
        body {
            font-family: 'Roboto Mono', monospace;
            background-color: #0E1117;
            color: #E0E0E0;
        }
        
        /* Map styles selector */
        .map-style-selector {
            margin-bottom: 10px;
            background-color: rgba(30, 30, 30, 0.7);
            border-radius: 5px;
            padding: 10px;
        }
        
        .map-style-selector h4 {
            margin-top: 0;
            margin-bottom: 5px;
            color: var(--accent-color, #FF4B4B);
        }
        
        /* Container for the map */
        .map-container {
            border: 2px solid var(--accent-color, #FF4B4B);
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
            background-color: rgba(30, 30, 30, 0.5);
        }
        
        .map-container h3 {
            margin-top: 0;
            margin-bottom: 10px;
            color: var(--accent-color, #FF4B4B);
        }
        
        .map-container iframe {
            width: 100%;
            height: 400px;
            border: none;
            background-color: white;
        }
        
        /* Comparison section */
        .comparison-section {
            border: 2px solid var(--accent-color, #FF4B4B);
            border-radius: 5px;
            padding: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
            background-color: rgba(30, 30, 30, 0.5);
        }
        
        .comparison-section h3 {
            margin-top: 0;
            margin-bottom: 10px;
            color: var(--accent-color, #FF4B4B);
        }
        
        /* Information boxes */
        .info-box {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: rgba(30, 30, 30, 0.5);
        }
        
        .info-box h3 {
            margin-top: 0;
            margin-bottom: 10px;
            color: var(--accent-color, #FF4B4B);
        }
        
        /* Chat container */
        .chat-container {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            margin-top: 20px;
            height: 300px;
            overflow-y: auto;
            padding: 10px;
            background-color: rgba(20, 20, 20, 0.8);
        }
        
        /* Chat bubbles */
        .chat-user, .chat-system {
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 5px;
            max-width: 85%;
        }
        
        .chat-user {
            background-color: #2C3E50;
            margin-left: auto;
            margin-right: 0;
            color: white;
            text-align: right;
        }
        
        .chat-system {
            background-color: #34495E;
            margin-left: 0;
            margin-right: auto;
            color: #E0E0E0;
        }
        
        /* Coordinates display */
        .coordinates {
            font-family: 'Roboto Mono', monospace;
            background-color: #1E1E1E;
            color: #4CAF50;
            padding: 5px 10px;
            border-radius: 3px;
            text-align: center;
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        /* Form inputs */
        .stTextInput input, .stTextArea textarea {
            background-color: #1E1E1E;
            color: #E0E0E0;
            border: 1px solid #444;
        }
        
        /* Buttons */
        .stButton button {
            background-color: var(--accent-color, #FF4B4B);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 3px;
            padding: 0.5rem 1rem;
            font-family: 'Roboto Mono', monospace;
        }
        
        .stButton button:hover {
            background-color: #1E3F20;
            color: var(--accent-color, #FF4B4B);
            border: 1px solid var(--accent-color, #FF4B4B);
        }
        
        /* Metrics section */
        .metrics-section {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        
        .metric-box {
            background-color: #1E1E1E;
            padding: 0.5rem;
            border-radius: 5px;
            flex: 1;
            margin: 0 0.5rem;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--accent-color, #FF4B4B);
        }
        
        .metric-label {
            font-size: 0.8rem;
            color: #888;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to create an interactive Mapbox map
def create_mapbox_map(latitude, longitude, zoom=13):
    """
    Create an interactive Mapbox map with custom styling.
    
    Args:
        latitude (float): The latitude coordinate
        longitude (float): The longitude coordinate
        zoom (int): The zoom level (1-18)
        
    Returns:
        str: HTML string for the interactive map
    """
    try:
        # Call the API to generate the map instead of creating it directly
        response = requests.post(
            f"{API_URL}/api/generate/interactive_map",
            json={"latitude": float(latitude), "longitude": float(longitude), "zoom": int(zoom)}
        )
        
        if response.status_code == 200:
            map_data = response.json()
            return map_data.get("map_html", "")
        else:
            print(f"Error calling map API: {response.text}")
            return ""
    except Exception as e:
        print(f"Error in create_mapbox_map: {str(e)}")
        return ""

# Main execution code at module level
if __name__ == "__main__":
    # Apply custom CSS first
    apply_custom_css()
    # Apply military style
    apply_military_style()
    # Initialize session state
    initialize_session_state() 