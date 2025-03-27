import exifread
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

class MetadataExtractor:
    """
    Utility class to extract metadata from images, including geolocation information.
    """
    
    @staticmethod
    def extract_metadata(image_path: str) -> Dict[str, Any]:
        """
        Extract all available metadata from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all metadata
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, 'rb') as image_file:
            tags = exifread.process_file(image_file, details=True)
            
        # Convert to a more manageable dictionary
        metadata = {
            "filename": os.path.basename(image_path),
            "file_size": os.path.getsize(image_path),
            "file_modified": datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat(),
            "exif_data": {str(k): str(v) for k, v in tags.items() if k not in ('JPEGThumbnail', 'TIFFThumbnail')}
        }
        
        # Extract GPS data if available
        gps_coords = MetadataExtractor.extract_gps_coordinates(tags)
        if gps_coords:
            metadata["gps_coordinates"] = gps_coords
            
        return metadata
    
    @staticmethod
    def extract_gps_coordinates(tags: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract GPS coordinates from EXIF tags
        
        Args:
            tags: EXIF tags extracted from an image
            
        Returns:
            Dictionary with latitude, longitude and reference data, or None if not available
        """
        # Check if GPS tags exist
        if not any(tag.startswith('GPS') for tag in tags):
            return None
            
        try:
            # Extract latitude
            lat_ref = str(tags.get('GPS GPSLatitudeRef', 'N'))
            lat = MetadataExtractor._convert_to_degrees(tags.get('GPS GPSLatitude'))
            if lat_ref == 'S':
                lat = -lat
                
            # Extract longitude
            lon_ref = str(tags.get('GPS GPSLongitudeRef', 'E'))
            lon = MetadataExtractor._convert_to_degrees(tags.get('GPS GPSLongitude'))
            if lon_ref == 'W':
                lon = -lon
                
            # Extract altitude if available
            altitude = None
            if 'GPS GPSAltitude' in tags:
                altitude_ref = tags.get('GPS GPSAltitudeRef', 0)
                altitude = float(str(tags['GPS GPSAltitude']).split('/')[0])
                if altitude_ref == 1:  # Below sea level
                    altitude = -altitude
                    
            # Return GPS data
            gps_data = {
                "latitude": lat,
                "longitude": lon,
                "latitude_ref": lat_ref,
                "longitude_ref": lon_ref,
            }
            
            if altitude is not None:
                gps_data["altitude"] = altitude
                
            return gps_data
        except (KeyError, ValueError, AttributeError, ZeroDivisionError) as e:
            print(f"Error extracting GPS data: {e}")
            return None
    
    @staticmethod
    def _convert_to_degrees(value) -> float:
        """
        Convert GPS coordinates stored in EXIF to decimal degrees
        
        Args:
            value: EXIF tag value containing degrees, minutes, seconds
            
        Returns:
            Decimal degrees as float
        """
        if not value:
            return 0.0
            
        degrees = float(str(value.values[0]))
        minutes = float(str(value.values[1]))
        seconds = float(str(value.values[2]))
        
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    
    @staticmethod
    def get_image_dimensions(metadata: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """
        Extract image dimensions from metadata
        
        Args:
            metadata: Image metadata dictionary
            
        Returns:
            Tuple with (width, height) or None if not available
        """
        exif = metadata.get('exif_data', {})
        
        # Try to get from standard EXIF tags
        if 'EXIF ExifImageWidth' in exif and 'EXIF ExifImageLength' in exif:
            try:
                width = int(str(exif['EXIF ExifImageWidth']).split(' ')[0])
                height = int(str(exif['EXIF ExifImageLength']).split(' ')[0])
                return (width, height)
            except (ValueError, IndexError):
                pass
                
        # Try alternative tags
        if 'Image ImageWidth' in exif and 'Image ImageLength' in exif:
            try:
                width = int(str(exif['Image ImageWidth']).split(' ')[0])
                height = int(str(exif['Image ImageLength']).split(' ')[0])
                return (width, height)
            except (ValueError, IndexError):
                pass
                
        return None 