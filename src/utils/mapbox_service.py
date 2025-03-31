import os
import requests
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import folium
from folium import plugins

# Load environment variables
load_dotenv()

class MapboxService:
    """
    Service for integrating with Mapbox APIs to get satellite imagery and static maps.
    """
    
    def __init__(self):
        """Initialize the Mapbox service with necessary API keys."""
        self.mapbox_api_key = os.getenv("MAPBOX_API_KEY", "")
        if not self.mapbox_api_key:
            print("WARNING: MAPBOX_API_KEY not found in environment variables")
    
    def get_satellite_image(self, latitude: float, longitude: float, zoom: int = 15, 
                           width: int = 600, height: int = 400) -> Optional[bytes]:
        """
        Get a satellite image for the specified coordinates using Mapbox Satellite API.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            zoom: Zoom level (0-22, higher is more detailed)
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Image data as bytes or None if error
        """
        try:
            # Build the Mapbox Static API URL for satellite imagery
            url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{longitude},{latitude},{zoom}/{width}x{height}?access_token={self.mapbox_api_key}"
            
            # Make the request
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            return response.content
            
        except Exception as e:
            print(f"Error getting satellite image: {str(e)}")
            return None
    
    def get_street_map(self, latitude: float, longitude: float, zoom: int = 15,
                     width: int = 600, height: int = 400) -> Optional[bytes]:
        """
        Get a street map for the specified coordinates using Mapbox Streets API.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            zoom: Zoom level (0-22)
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Image data as bytes or None if error
        """
        try:
            # Build the Mapbox Static API URL for street map
            url = f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/static/{longitude},{latitude},{zoom}/{width}x{height}?access_token={self.mapbox_api_key}"
            
            # Make the request
            response = requests.get(url)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            print(f"Error getting street map: {str(e)}")
            return None
    
    def get_terrain_map(self, latitude: float, longitude: float, zoom: int = 15,
                     width: int = 600, height: int = 400) -> Optional[bytes]:
        """
        Get a terrain map for the specified coordinates using Mapbox Outdoors style.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            zoom: Zoom level (0-22)
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Image data as bytes or None if error
        """
        try:
            # Build the Mapbox Static API URL for outdoor/terrain map
            url = f"https://api.mapbox.com/styles/v1/mapbox/outdoors-v11/static/{longitude},{latitude},{zoom}/{width}x{height}?access_token={self.mapbox_api_key}"
            
            # Make the request
            response = requests.get(url)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            print(f"Error getting terrain map: {str(e)}")
            return None
    
    def generate_comparison_html(self, drone_image_path: str, latitude: float, longitude: float) -> str:
        """
        Generate HTML for a comparison view between drone image and maps.
        
        Args:
            drone_image_path: Path to the drone image
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            
        Returns:
            HTML string for the comparison
        """
        try:
            # Get the satellite and map images
            satellite_image = self.get_satellite_image(latitude, longitude)
            street_map = self.get_street_map(latitude, longitude)
            terrain_map = self.get_terrain_map(latitude, longitude)
            
            # Drone image
            try:
                with open(drone_image_path, 'rb') as f:
                    drone_image_data = f.read()
                    drone_b64 = base64.b64encode(drone_image_data).decode('utf-8')
            except Exception as e:
                print(f"Error reading drone image: {str(e)}")
                drone_b64 = None
            
            # Satellite image
            satellite_b64 = None
            if satellite_image:
                try:
                    satellite_b64 = base64.b64encode(satellite_image).decode('utf-8')
                except Exception as e:
                    print(f"Error encoding satellite image: {str(e)}")
            
            # Street map
            street_b64 = None
            if street_map:
                try:
                    street_b64 = base64.b64encode(street_map).decode('utf-8')
                except Exception as e:
                    print(f"Error encoding street map: {str(e)}")
            
            # Terrain map
            terrain_b64 = None
            if terrain_map:
                try:
                    terrain_b64 = base64.b64encode(terrain_map).decode('utf-8')
                except Exception as e:
                    print(f"Error encoding terrain map: {str(e)}")
            
            # Create HTML
            html = """
            <div style="display: flex; flex-direction: column; gap: 10px; width: 100%;">
                <div style="display: flex; flex-direction: row; gap: 10px; width: 100%;">
                    <div style="flex: 1; border: 1px solid #ccc; border-radius: 5px; padding: 10px; background-color: #f8f8f8;">
                        <h4 style="margin-top: 0; text-align: center; color: #444;">Drone Image</h4>
            """
            
            if drone_b64:
                html += f'<img src="data:image/jpeg;base64,{drone_b64}" alt="Drone Image" style="width: 100%; height: auto; max-height: 300px; object-fit: cover; border-radius: 3px;" />'
            else:
                html += '<p>Drone image not available</p>'
            
            html += """
                    </div>
                    <div style="flex: 1; border: 1px solid #ccc; border-radius: 5px; padding: 10px; background-color: #f8f8f8;">
                        <h4 style="margin-top: 0; text-align: center; color: #444;">Satellite View</h4>
            """
            
            if satellite_b64:
                html += f'<img src="data:image/png;base64,{satellite_b64}" alt="Satellite Image" style="width: 100%; height: auto; max-height: 300px; object-fit: cover; border-radius: 3px;" />'
            else:
                html += '<p>Satellite image not available</p>'
            
            html += """
                    </div>
                </div>
                <div style="display: flex; flex-direction: row; gap: 10px; width: 100%;">
                    <div style="flex: 1; border: 1px solid #ccc; border-radius: 5px; padding: 10px; background-color: #f8f8f8;">
                        <h4 style="margin-top: 0; text-align: center; color: #444;">Street Map</h4>
            """
            
            if street_b64:
                html += f'<img src="data:image/png;base64,{street_b64}" alt="Street Map" style="width: 100%; height: auto; max-height: 300px; object-fit: cover; border-radius: 3px;" />'
            else:
                html += '<p>Street map not available</p>'
            
            html += """
                    </div>
                    <div style="flex: 1; border: 1px solid #ccc; border-radius: 5px; padding: 10px; background-color: #f8f8f8;">
                        <h4 style="margin-top: 0; text-align: center; color: #444;">Terrain Map</h4>
            """
            
            if terrain_b64:
                html += f'<img src="data:image/png;base64,{terrain_b64}" alt="Terrain Map" style="width: 100%; height: auto; max-height: 300px; object-fit: cover; border-radius: 3px;" />'
            else:
                html += '<p>Terrain map not available</p>'
            
            html += """
                    </div>
                </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            print(f"Error generating comparison HTML: {str(e)}")
            return f"<div>Error generating comparison view: {str(e)}</div>"
    
    def generate_interactive_map(self, latitude: float, longitude: float, zoom: int = 15) -> str:
        """
        Generate an interactive map with Mapbox tiles.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            zoom: Zoom level
            
        Returns:
            HTML string containing the interactive map
        """
        try:
            # Create a map centered at the specified location
            m = folium.Map(
                location=[latitude, longitude],
                zoom_start=zoom,
                width='100%',
                height='400px',
                tiles='https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v11/tiles/{z}/{x}/{y}?access_token=' + self.mapbox_api_key,
                attr='Mapbox'
            )
            
            # Add layer control to switch between map styles
            folium.TileLayer(
                tiles='https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{z}/{x}/{y}?access_token=' + self.mapbox_api_key,
                attr='Mapbox Streets',
                name='Streets'
            ).add_to(m)
            
            folium.TileLayer(
                tiles='https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=' + self.mapbox_api_key,
                attr='Mapbox Satellite',
                name='Satellite'
            ).add_to(m)
            
            folium.TileLayer(
                tiles='https://api.mapbox.com/styles/v1/mapbox/outdoors-v11/tiles/{z}/{x}/{y}?access_token=' + self.mapbox_api_key,
                attr='Mapbox Outdoors',
                name='Terrain'
            ).add_to(m)
            
            # Add a marker
            folium.Marker(
                [latitude, longitude],
                popup=f"Lat: {latitude:.6f}, Long: {longitude:.6f}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            
            # Add a circle to represent approximate area
            folium.Circle(
                radius=50,
                location=[latitude, longitude],
                popup='Approximate Area',
                color="crimson", 
                fill=True,
                fill_color="crimson",
                fill_opacity=0.2
            ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Get the HTML
            html = m._repr_html_()
            
            return html
            
        except Exception as e:
            print(f"Error generating interactive map: {str(e)}")
            return f"<div>Error generating interactive map: {str(e)}</div>" 