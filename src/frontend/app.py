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
            padding: 5px;
        }
        
        /* Chat messages */
        .chat-container {
            border: 1px solid var(--accent-yellow);
            border-radius: 5px;
            padding: 10px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .user-message {
            background-color: var(--military-green);
            color: var(--text-color);
            padding: 8px;
            border-radius: 5px;
            margin-bottom: 8px;
            text-align: right;
        }
        
        .assistant-message {
            background-color: rgba(30, 63, 32, 0.7);
            color: var(--text-color);
            padding: 8px;
            border-radius: 5px;
            margin-bottom: 8px;
            border-left: 3px solid var(--accent-yellow);
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

# Initialize session state
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
    st.markdown("**MODEL**: GPT-4 Vision")
    st.markdown("**ACCURACY**: HIGH")
    st.markdown("**RESPONSE TIME**: 2-5s")
    st.markdown("</div>", unsafe_allow_html=True)

# Main content area
if st.session_state.active_tab == "upload":
    st.markdown("## 👁️ TERRAIN INTELLIGENCE ANALYSIS")
    
    # File uploader
    uploaded_file = st.file_uploader("UPLOAD RECONNAISSANCE IMAGE", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image(image, use_column_width=True)
        
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
                with st.spinner("PROCESSING INTELLIGENCE DATA..."):
                    # Save the image to a BytesIO object
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    # Upload to the API
                    files = {"file": (uploaded_file.name, byte_im, "image/jpeg")}
                    response = requests.post(f"{API_URL}/api/upload/image", files=files)
                    
                    if response.status_code == 200:
                        upload_data = response.json()
                        image_id = upload_data.get("image_id")
                        st.session_state.current_image_id = image_id
                        
                        # Analyze the image
                        analysis_response = requests.post(f"{API_URL}/api/analyze/image/{image_id}")
                        
                        if analysis_response.status_code == 200:
                            analysis_data = analysis_response.json()
                            st.session_state.analysis_results = analysis_data
                            
                            # Print debug info
                            print(f"Analysis data keys: {', '.join(analysis_data.keys())}")
                            if "llm_analysis" in analysis_data:
                                print(f"LLM analysis keys: {', '.join(analysis_data['llm_analysis'].keys())}")
                            if "geo_data" in analysis_data:
                                print(f"Geo data keys: {', '.join(analysis_data['geo_data'].keys())}")
                            
                            # Extract map HTML if available
                            if analysis_data.get("geo_data") and "map" in analysis_data["geo_data"]:
                                st.session_state.map_html = analysis_data["geo_data"]["map"]
                            
                            st.success("ANALYSIS COMPLETE")
                        else:
                            st.error(f"ERROR ANALYZING IMAGE: {analysis_response.text}")
                    else:
                        st.error(f"ERROR UPLOADING IMAGE: {response.text}")
    
    # Display analysis results
    if st.session_state.analysis_results:
        st.markdown("## 📊 INTELLIGENCE REPORT")
        
        # Mostrar el análisis en bruto en caso de que la visualización falle
        with st.expander("Raw Analysis Data (Debug)", expanded=False):
            st.json(st.session_state.analysis_results)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Map
            if st.session_state.map_html:
                st.markdown("<div class='map-container'>", unsafe_allow_html=True)
                st.markdown("<h3>GEOLOCATION DATA</h3>", unsafe_allow_html=True)
                st.markdown(st.session_state.map_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Location details
            if "geo_data" in st.session_state.analysis_results and st.session_state.analysis_results["geo_data"]:
                geo_data = st.session_state.analysis_results["geo_data"]
                
                # Mostrar texto del análisis si está disponible
                if "text_analysis" in geo_data:
                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                    st.markdown("### GEO ANALYSIS")
                    st.markdown(geo_data["text_analysis"])
                    st.markdown("</div>", unsafe_allow_html=True)
                
                if "merged_data" in geo_data:
                    merged_data = geo_data["merged_data"]
                    
                    if "coordinates" in merged_data:
                        coords = merged_data["coordinates"]
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
                    
                    if "address" in merged_data:
                        address = merged_data["address"]
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        st.markdown("### LOCATION DETAILS")
                        st.markdown(f"**COUNTRY**: {address.get('country', 'Unknown')}")
                        st.markdown(f"**CITY**: {address.get('city', 'Unknown')}")
                        st.markdown(f"**DISTRICT**: {address.get('district', 'Unknown')}")
                        st.markdown(f"**NEIGHBORHOOD**: {address.get('neighborhood', 'Unknown')}")
                        st.markdown(f"**STREET**: {address.get('street', 'Unknown')}")
                        st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Mostrar el análisis completo si los demás componentes fallan
            if "llm_analysis" in st.session_state.analysis_results and "analysis" in st.session_state.analysis_results["llm_analysis"]:
                analysis = st.session_state.analysis_results["llm_analysis"]["analysis"]
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown("### FULL ANALYSIS")
                st.markdown(analysis)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Confidence levels
            if "llm_analysis" in st.session_state.analysis_results and "confidence_level" in st.session_state.analysis_results["llm_analysis"]:
                confidence = st.session_state.analysis_results["llm_analysis"]["confidence_level"]
                
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown("### CONFIDENCE ASSESSMENT")
                
                # Show text confidence values as fallback
                st.markdown(f"**Overall**: {confidence.get('overall', 'Unknown')}")
                st.markdown(f"**Country**: {confidence.get('country', 'Unknown')}")
                st.markdown(f"**City**: {confidence.get('city', 'Unknown')}")
                st.markdown(f"**District**: {confidence.get('district', 'Unknown')}")
                st.markdown(f"**Street**: {confidence.get('street', 'Unknown')}")
                st.markdown(f"**Coordinates**: {confidence.get('coordinates', 'Unknown')}")
                
                try:
                    # Create radar chart for confidence levels
                    confidence_values = {
                        "Overall": 0,
                        "Country": 0,
                        "City": 0,
                        "District": 0,
                        "Neighborhood": 0,
                        "Street": 0,
                        "Coordinates": 0
                    }
                    
                    # Map text confidence to numeric values
                    confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
                    
                    if "overall" in confidence:
                        confidence_values["Overall"] = confidence_map.get(confidence["overall"].lower(), 0)
                    if "country" in confidence:
                        confidence_values["Country"] = confidence_map.get(confidence["country"].lower(), 0)
                    if "city" in confidence:
                        confidence_values["City"] = confidence_map.get(confidence["city"].lower(), 0)
                    if "district" in confidence:
                        confidence_values["District"] = confidence_map.get(confidence["district"].lower(), 0)
                    if "neighborhood" in confidence:
                        confidence_values["Neighborhood"] = confidence_map.get(confidence["neighborhood"].lower(), 0)
                    if "street" in confidence:
                        confidence_values["Street"] = confidence_map.get(confidence["street"].lower(), 0)
                    if "coordinates" in confidence:
                        confidence_values["Coordinates"] = confidence_map.get(confidence["coordinates"].lower(), 0)
                    
                    categories = list(confidence_values.keys())
                    values = list(confidence_values.values())
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        fillcolor='rgba(30, 63, 32, 0.7)',
                        line=dict(color='#DAA520')
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            ),
                            bgcolor='rgba(15, 28, 46, 0.7)'
                        ),
                        paper_bgcolor='rgba(15, 28, 46, 0)',
                        plot_bgcolor='rgba(15, 28, 46, 0)',
                        font=dict(color='#E0E0E0'),
                        height=350,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating confidence chart: {str(e)}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Evidence
            if "llm_analysis" in st.session_state.analysis_results and "evidence" in st.session_state.analysis_results["llm_analysis"]:
                evidence = st.session_state.analysis_results["llm_analysis"]["evidence"]
                
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown("### INTELLIGENCE MARKERS")
                
                if "landmarks" in evidence and evidence["landmarks"]:
                    st.markdown("**LANDMARKS:**")
                    for landmark in evidence["landmarks"]:
                        st.markdown(f"- {landmark}")
                
                if "terrain_features" in evidence and evidence["terrain_features"]:
                    st.markdown("**TERRAIN FEATURES:**")
                    for feature in evidence["terrain_features"]:
                        st.markdown(f"- {feature}")
                
                if "architectural_elements" in evidence and evidence["architectural_elements"]:
                    st.markdown("**ARCHITECTURAL ELEMENTS:**")
                    for element in evidence["architectural_elements"]:
                        st.markdown(f"- {element}")
                
                if "signage" in evidence and evidence["signage"]:
                    st.markdown("**SIGNAGE:**")
                    for sign in evidence["signage"]:
                        st.markdown(f"- {sign}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Reasoning
            if "llm_analysis" in st.session_state.analysis_results and "reasoning" in st.session_state.analysis_results["llm_analysis"]:
                reasoning = st.session_state.analysis_results["llm_analysis"]["reasoning"]
                
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.markdown("### ANALYSIS RATIONALE")
                st.markdown(reasoning)
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
                    st.markdown(st.session_state.map_html, unsafe_allow_html=True)
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
                    st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
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
                            
                            # Add assistant response to chat history
                            assistant_response = chat_data.get("response", "ERROR: No response received")
                            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                            
                            # Clear the input field (need to rerun the app)
                            st.experimental_rerun()
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