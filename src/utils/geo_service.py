import os
import requests
from typing import Dict, Any, Optional, Tuple
from geopy.geocoders import Nominatim
import folium
from dotenv import load_dotenv
import math

load_dotenv()

class GeoService:
    """
    Service for handling geolocation processing, verification, and mapping.
    """
    
    def __init__(self):
        """Initialize the geolocation service with necessary API keys and services."""
        self.google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
        # Initialize Nominatim for reverse geocoding (no API key needed)
        self.geolocator = Nominatim(user_agent="drone-osint-geospy")
    
    def get_location_from_coordinates(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Get detailed location information from coordinates using reverse geocoding.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            
        Returns:
            Dictionary with detailed location information
        """
        try:
            # Use Nominatim for reverse geocoding
            location = self.geolocator.reverse((latitude, longitude), language="en")
            
            if not location:
                return {"error": "Location not found"}
                
            address_data = location.raw.get('address', {})
            
            # Structure the response
            location_data = {
                "coordinates": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "address": {
                    "country": address_data.get("country", "Unknown"),
                    "country_code": address_data.get("country_code", "").upper(),
                    "state": address_data.get("state", address_data.get("region", "Unknown")),
                    "county": address_data.get("county", "Unknown"),
                    "city": address_data.get("city", address_data.get("town", address_data.get("village", "Unknown"))),
                    "district": address_data.get("suburb", address_data.get("district", "Unknown")),
                    "neighborhood": address_data.get("neighbourhood", address_data.get("hamlet", "Unknown")),
                    "street": address_data.get("road", address_data.get("street", "Unknown")),
                    "postal_code": address_data.get("postcode", "Unknown"),
                },
                "display_name": location.address,
                "raw_data": address_data
            }
            
            return location_data
            
        except Exception as e:
            return {"error": f"Error getting location data: {str(e)}"}
    
    def enhance_with_google_maps(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Enhance location data with Google Maps API (if API key is provided).
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            
        Returns:
            Enhanced location data or error message
        """
        if not self.google_maps_api_key:
            return {"error": "Google Maps API key not configured"}
            
        try:
            # Google Maps Geocoding API endpoint
            url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={self.google_maps_api_key}"
            response = requests.get(url)
            data = response.json()
            
            if data.get("status") != "OK":
                return {"error": f"Google Maps API error: {data.get('status')}"}
                
            results = data.get("results", [])
            if not results:
                return {"error": "No results from Google Maps API"}
                
            # Process the first result (most accurate)
            result = results[0]
            address_components = result.get("address_components", [])
            
            # Structure the response
            location_data = {
                "coordinates": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "google_maps": {
                    "formatted_address": result.get("formatted_address", "Unknown"),
                    "place_id": result.get("place_id", "Unknown"),
                    "components": {}
                }
            }
            
            # Process address components
            for component in address_components:
                types = component.get("types", [])
                if types:
                    component_type = types[0]
                    location_data["google_maps"]["components"][component_type] = {
                        "long_name": component.get("long_name", ""),
                        "short_name": component.get("short_name", "")
                    }
            
            return location_data
            
        except Exception as e:
            return {"error": f"Error with Google Maps API: {str(e)}"}
    
    def generate_map(self, latitude: float, longitude: float, zoom: int = 15) -> str:
        """
        Generate a folium map centered on the given coordinates.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            zoom: Zoom level for the map
            
        Returns:
            HTML string containing the map
        """
        try:
            # Create a map centered at the specified location
            map_obj = folium.Map(
                location=[latitude, longitude],
                zoom_start=zoom,
                width='100%',
                height='400px',
                control_scale=True
            )
            
            # Add a marker at the exact location
            folium.Marker(
                [latitude, longitude],
                popup=f"Lat: {latitude:.6f}, Long: {longitude:.6f}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(map_obj)
            
            # Add a circle to represent approximate accuracy
            folium.Circle(
                radius=50,
                location=[latitude, longitude],
                popup='Approximate Area',
                color="crimson",
                fill=True,
                fill_color="crimson",
                fill_opacity=0.2
            ).add_to(map_obj)
            
            # Get the HTML representation
            html = map_obj._repr_html_()
            
            # Clean up the HTML
            html = html.replace('\n', '').replace('  ', '')
            
            return html
            
        except Exception as e:
            print(f"Error generating map: {str(e)}")
            return None
    
    def merge_location_data(self, llm_data: Dict[str, Any], metadata_gps: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge and validate location data from multiple sources (Vision LLM and image metadata).
        
        Args:
            llm_data: Location data extracted by the vision LLM
            metadata_gps: GPS coordinates from image metadata
            
        Returns:
            Merged and validated location data
        """
        result = {
            "sources": {},
            "merged_data": {},
            "confidence": {}
        }
        
        # Add LLM data if available
        if llm_data and isinstance(llm_data, dict):
            result["sources"]["llm"] = llm_data
            
            # Extract coordinates if available in LLM data
            llm_coords = llm_data.get("coordinates", {})
            if llm_coords and "latitude" in llm_coords and "longitude" in llm_coords:
                result["merged_data"]["coordinates"] = llm_coords
                result["confidence"]["coordinates_source"] = "LLM"
            
            # Extract address information
            llm_address = llm_data.get("address", {})
            if llm_address:
                result["merged_data"]["address"] = llm_address
                result["confidence"]["address_source"] = "LLM"
        
        # Add metadata GPS if available
        if metadata_gps and isinstance(metadata_gps, dict):
            result["sources"]["metadata"] = {
                "coordinates": {
                    "latitude": metadata_gps.get("latitude"),
                    "longitude": metadata_gps.get("longitude")
                }
            }
            
            # If we don't have coordinates from LLM, use metadata
            if "coordinates" not in result["merged_data"]:
                result["merged_data"]["coordinates"] = result["sources"]["metadata"]["coordinates"]
                result["confidence"]["coordinates_source"] = "Metadata"
            
            # If we have coordinates from both sources, calculate the difference
            elif "coordinates" in result["merged_data"]:
                lat_diff = abs(metadata_gps.get("latitude", 0) - result["merged_data"]["coordinates"].get("latitude", 0))
                lon_diff = abs(metadata_gps.get("longitude", 0) - result["merged_data"]["coordinates"].get("longitude", 0))
                
                result["confidence"]["coordinate_difference"] = {
                    "latitude_diff": lat_diff,
                    "longitude_diff": lon_diff,
                    "difference_meters": self._calculate_distance(
                        result["merged_data"]["coordinates"].get("latitude", 0), 
                        result["merged_data"]["coordinates"].get("longitude", 0),
                        metadata_gps.get("latitude", 0), 
                        metadata_gps.get("longitude", 0)
                    )
                }
        
        # If we have coordinates, enhance with reverse geocoding if address is not available
        if "coordinates" in result["merged_data"] and "address" not in result["merged_data"]:
            try:
                coords = result["merged_data"]["coordinates"]
                location_data = self.get_location_from_coordinates(coords["latitude"], coords["longitude"])
                
                if "address" in location_data and "error" not in location_data:
                    result["merged_data"]["address"] = location_data["address"]
                    result["confidence"]["address_source"] = "Reverse Geocoding"
                    result["sources"]["reverse_geocoding"] = location_data
            except Exception as e:
                result["errors"] = [f"Error in reverse geocoding: {str(e)}"]
        
        return result
    
    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the distance between two coordinates in kilometers."""
        # Approximate radius of Earth in km
        R = 6371
        
        # Convert latitudes and longitudes from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Differences
        d_lat = lat2_rad - lat1_rad
        d_lon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = math.sin(d_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def generate_interactive_map(self, latitude: float, longitude: float, zoom: int = 15) -> str:
        """
        Generate an interactive map for the given coordinates using folium.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            zoom: Zoom level (1-18)
            
        Returns:
            HTML string for the interactive map
        """
        try:
            # Create a new map centered at the specified coordinates
            m = folium.Map(location=[latitude, longitude], zoom_start=zoom, 
                          tiles='https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/256/{z}/{x}/{y}@2x?access_token=' + self.mapbox_token,
                          attr='Mapbox')
            
            # Add a marker at the coordinates
            tooltip = f"Lat: {latitude}, Lon: {longitude}"
            folium.Marker(
                [latitude, longitude], 
                tooltip=tooltip,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add a circle to indicate a radius
            folium.Circle(
                radius=500,  # 500 meters
                location=[latitude, longitude],
                color='crimson',
                fill=True,
                fill_color='crimson',
                fill_opacity=0.2
            ).add_to(m)
            
            # Add LayerControl to switch between different map styles
            folium.TileLayer(
                tiles='https://api.mapbox.com/styles/v1/mapbox/streets-v12/tiles/256/{z}/{x}/{y}@2x?access_token=' + self.mapbox_token,
                attr='Mapbox Streets',
                name='Streets'
            ).add_to(m)
            
            folium.TileLayer(
                tiles='https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{z}/{x}/{y}@2x?access_token=' + self.mapbox_token,
                attr='Mapbox Satellite',
                name='Satellite'
            ).add_to(m)
            
            folium.TileLayer(
                tiles='https://api.mapbox.com/styles/v1/mapbox/outdoors-v12/tiles/256/{z}/{x}/{y}@2x?access_token=' + self.mapbox_token,
                attr='Mapbox Outdoors',
                name='Outdoors'
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            # Get the HTML representation
            html_map = m._repr_html_()
            
            return html_map
            
        except Exception as e:
            print(f"Error generating interactive map: {str(e)}")
            # Fallback to a basic map
            try:
                m = folium.Map(location=[latitude, longitude], zoom_start=zoom)
                folium.Marker([latitude, longitude]).add_to(m)
                html_map = m._repr_html_()
                return html_map
            except Exception as fallback_err:
                print(f"Fallback map generation failed: {str(fallback_err)}")
                return "" 