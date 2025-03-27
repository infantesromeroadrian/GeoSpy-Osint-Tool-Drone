import os
import requests
from typing import Dict, Any, Optional, Tuple
from geopy.geocoders import Nominatim
import folium
from dotenv import load_dotenv

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
            map_obj = folium.Map(location=[latitude, longitude], zoom_start=zoom)
            
            # Add a marker at the exact location
            folium.Marker(
                [latitude, longitude],
                popup=f"Lat: {latitude:.6f}, Long: {longitude:.6f}",
                icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
            ).add_to(map_obj)
            
            # Add a circle to represent approximate accuracy
            folium.Circle(
                radius=50,
                location=[latitude, longitude],
                color="crimson",
                fill=True,
                fill_color="crimson",
                fill_opacity=0.2
            ).add_to(map_obj)
            
            # Return the HTML representation of the map
            return map_obj._repr_html_()
            
        except Exception as e:
            return f"<div class='error'>Error generating map: {str(e)}</div>"
    
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
        """
        Calculate the Haversine distance between two points in meters.
        
        Args:
            lat1: Latitude of point 1 in decimal degrees
            lon1: Longitude of point 1 in decimal degrees
            lat2: Latitude of point 2 in decimal degrees
            lon2: Longitude of point 2 in decimal degrees
            
        Returns:
            Distance in meters
        """
        from math import radians, sin, cos, sqrt, atan2
        
        # Earth radius in meters
        R = 6371000
        
        # Convert coordinates to radians
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)
        
        # Calculate differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        return distance 