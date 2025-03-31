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
    
    def geocode_forward(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform forward geocoding (address to coordinates) using Mapbox Geocoding API.
        
        Args:
            query: The address or place name to geocode
            limit: Maximum number of results to return (1-10)
            
        Returns:
            List of geocoding results with coordinates and other data
        """
        try:
            # Ensure limit is between 1 and 10
            limit = max(1, min(10, limit))
            
            # Build the Mapbox Geocoding API URL
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json?access_token={self.mapbox_api_key}&limit={limit}"
            
            # Make the request
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Extract and format the results
            results = []
            for feature in data.get("features", []):
                center = feature.get("center", [0, 0])
                result = {
                    "name": feature.get("text", ""),
                    "place_name": feature.get("place_name", ""),
                    "longitude": center[0],
                    "latitude": center[1],
                    "place_type": feature.get("place_type", []),
                    "relevance": feature.get("relevance", 0),
                    "bbox": feature.get("bbox")
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in forward geocoding: {str(e)}")
            return []
    
    def geocode_reverse(self, longitude: float, latitude: float) -> Dict[str, Any]:
        """
        Perform reverse geocoding (coordinates to address) using Mapbox Geocoding API.
        
        Args:
            longitude: Longitude in decimal degrees
            latitude: Latitude in decimal degrees
            
        Returns:
            Dictionary with address data
        """
        try:
            # Build the Mapbox Geocoding API URL
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{longitude},{latitude}.json?access_token={self.mapbox_api_key}"
            
            # Make the request
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Extract the most relevant result
            if data.get("features") and len(data["features"]) > 0:
                feature = data["features"][0]
                center = feature.get("center", [0, 0])
                
                # Extract address components
                address_components = {}
                for context in feature.get("context", []):
                    context_id = context.get("id", "")
                    if context_id.startswith("country"):
                        address_components["country"] = context.get("text", "")
                    elif context_id.startswith("region"):
                        address_components["region"] = context.get("text", "")
                    elif context_id.startswith("district"):
                        address_components["district"] = context.get("text", "")
                    elif context_id.startswith("place"):
                        address_components["city"] = context.get("text", "")
                    elif context_id.startswith("postcode"):
                        address_components["postcode"] = context.get("text", "")
                    elif context_id.startswith("neighborhood"):
                        address_components["neighborhood"] = context.get("text", "")
                
                # Create the result
                result = {
                    "place_name": feature.get("place_name", ""),
                    "longitude": center[0],
                    "latitude": center[1],
                    "address_components": address_components,
                    "address_type": feature.get("place_type", []),
                }
                
                return result
            
            return {"error": "No results found"}
            
        except Exception as e:
            print(f"Error in reverse geocoding: {str(e)}")
            return {"error": str(e)}
    
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
    
    def get_static_map(self, latitude: float, longitude: float, zoom: int = 15,
                      width: int = 600, height: int = 400, style: str = "streets-v11",
                      marker: bool = True) -> Optional[bytes]:
        """
        Get a static map image using Mapbox Static Images API.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            zoom: Zoom level (0-22)
            width: Image width in pixels
            height: Image height in pixels
            style: Map style (e.g., streets-v11, dark-v10, satellite-v9)
            marker: Whether to include a marker at the specified coordinates
            
        Returns:
            Image data as bytes or None if error
        """
        try:
            # Initialize URL parts
            url_parts = [
                f"https://api.mapbox.com/styles/v1/mapbox/{style}/static"
            ]
            
            # Add marker if requested
            if marker:
                url_parts.append(f"pin-s+FF4B4B({longitude},{latitude})")
            
            # Add center coordinates and zoom
            url_parts.append(f"{longitude},{latitude},{zoom}")
            
            # Add dimensions and access token
            url_parts.append(f"{width}x{height}")
            url_parts.append(f"?access_token={self.mapbox_api_key}")
            
            # Build the complete URL
            if marker:
                url = f"{url_parts[0]}/{url_parts[1]},{url_parts[2]}/{url_parts[3]}{url_parts[4]}"
            else:
                url = f"{url_parts[0]}/{url_parts[2]}/{url_parts[3]}{url_parts[4]}"
            
            # Make the request
            response = requests.get(url)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            print(f"Error getting static map: {str(e)}")
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
            url = f"https://api.mapbox.com/styles/v1/mapbox/streets-v11/static/pin-s+FF4B4B({longitude},{latitude})/{longitude},{latitude},{zoom}/{width}x{height}?access_token={self.mapbox_api_key}"
            
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
            # Build the Mapbox Static API URL for terrain map
            url = f"https://api.mapbox.com/styles/v1/mapbox/outdoors-v11/static/pin-s+FF4B4B({longitude},{latitude})/{longitude},{latitude},{zoom}/{width}x{height}?access_token={self.mapbox_api_key}"
            
            # Make the request
            response = requests.get(url)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            print(f"Error getting terrain map: {str(e)}")
            return None
    
    def generate_comparison_html(self, image_path: str, latitude: float, longitude: float, 
                              zoom: int = 15, width: int = 500, height: int = 500) -> str:
        """
        Generate HTML for displaying a comparison between an image and maps.
        
        Args:
            image_path: Path to the image file to compare
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            zoom: Zoom level for the maps
            width: Width of each image in the comparison
            height: Height of each image in the comparison
            
        Returns:
            HTML string with the comparison layout
        """
        try:
            # Get maps at different styles
            satellite_img = self.get_satellite_image(latitude, longitude, zoom, width, height)
            street_img = self.get_street_map(latitude, longitude, zoom, width, height)
            terrain_img = self.get_terrain_map(latitude, longitude, zoom, width, height)
            
            # Read the drone image
            with open(image_path, "rb") as img_file:
                drone_img = img_file.read()
                
            # Convert images to base64 for embedding in HTML
            satellite_b64 = base64.b64encode(satellite_img).decode('utf-8') if satellite_img else ""
            street_b64 = base64.b64encode(street_img).decode('utf-8') if street_img else ""
            terrain_b64 = base64.b64encode(terrain_img).decode('utf-8') if terrain_img else ""
            drone_b64 = base64.b64encode(drone_img).decode('utf-8')
            
            # Create HTML
            html = f"""
            <div style="display: flex; flex-direction: column; font-family: Arial, sans-serif;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                    <div style="width: {width}px;">
                        <h3 style="text-align: center;">Drone Image</h3>
                        <img src="data:image/jpeg;base64,{drone_b64}" style="width: 100%; border: 2px solid #333; border-radius: 5px;" />
                        <p style="text-align: center;">COORDINATES: {latitude:.6f}, {longitude:.6f}</p>
                    </div>
                    <div style="width: {width}px;">
                        <h3 style="text-align: center;">Satellite View</h3>
                        {f'<img src="data:image/png;base64,{satellite_b64}" style="width: 100%; border: 2px solid #333; border-radius: 5px;" />' if satellite_b64 else '<div style="width: 100%; height: {height}px; display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><p>Satellite image not available</p></div>'}
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <div style="width: {width}px;">
                        <h3 style="text-align: center;">Street Map</h3>
                        {f'<img src="data:image/png;base64,{street_b64}" style="width: 100%; border: 2px solid #333; border-radius: 5px;" />' if street_b64 else '<div style="width: 100%; height: {height}px; display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><p>Street map not available</p></div>'}
                    </div>
                    <div style="width: {width}px;">
                        <h3 style="text-align: center;">Terrain Map</h3>
                        {f'<img src="data:image/png;base64,{terrain_b64}" style="width: 100%; border: 2px solid #333; border-radius: 5px;" />' if terrain_b64 else '<div style="width: 100%; height: {height}px; display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><p>Terrain map not available</p></div>'}
                    </div>
                </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            print(f"Error generating comparison HTML: {str(e)}")
            error_html = f"""
            <div style="padding: 20px; background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; border-radius: 5px;">
                <h3>Error Generating Comparison</h3>
                <p>{str(e)}</p>
                <p>Coordinates: {latitude:.6f}, {longitude:.6f}</p>
            </div>
            """
            return error_html
    
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