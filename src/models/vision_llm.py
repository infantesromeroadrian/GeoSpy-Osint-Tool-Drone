import os
import json
import base64
from typing import Dict, Any, List, Optional, Union
import time
import datetime
import sys
import logging
import io
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class VisionLLM:
    """
    Class to interact with Vision-capable LLMs (like OpenAI's GPT-4V) to analyze images
    and extract geolocation and terrain information.
    """
    
    def __init__(self):
        """Initialize the Vision LLM with Gemini API."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
        logger.info("Gemini client initialized successfully")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using Gemini Vision."""
        try:
            # Load and prepare the image
            image = Image.open(image_path)
            
            # Prepare the prompt
            prompt = """Analiza esta imagen y proporciona la siguiente información en formato JSON:
            {
                "description": "Descripción detallada del entorno físico",
                "location": {
                    "country": "País probable",
                    "city": "Ciudad probable",
                    "neighborhood": "Barrio o área específica",
                    "street": "Calle o ubicación específica",
                    "coordinates": {
                        "latitude": "Latitud aproximada",
                        "longitude": "Longitud aproximada"
                    }
                },
                "architectural_features": [
                    "Lista de características arquitectónicas distintivas"
                ],
                "landscape_features": [
                    "Lista de características del paisaje"
                ],
                "confidence": "Nivel de confianza en la geolocalización (alto/medio/bajo)"
            }
            
            IMPORTANTE: 
            - Enfócate SOLO en elementos arquitectónicos y geográficos
            - NO incluyas análisis de personas
            - Proporciona coordenadas aproximadas basadas en características visibles
            - Indica el nivel de confianza en la geolocalización"""

            # Generate response
            response = self.model.generate_content([prompt, image])
            
            # Parse the response
            try:
                # Extract JSON from the response
                response_text = response.text
                # Find the JSON part in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    # Parse the JSON
                    import json
                    analysis_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except Exception as e:
                logger.error(f"Error parsing Gemini response: {str(e)}")
                analysis_result = {
                    "description": response_text,
                    "location": {
                        "country": "Unknown",
                        "city": "Unknown",
                        "neighborhood": "Unknown",
                        "street": "Unknown",
                        "coordinates": {
                            "latitude": "0",
                            "longitude": "0"
                        }
                    },
                    "architectural_features": [],
                    "landscape_features": [],
                    "confidence": "low"
                }

            return {
                "llm_analysis": {
                    "description": analysis_result.get("description", ""),
                    "confidence": analysis_result.get("confidence", "low")
                },
                "geo_data": {
                    "country": analysis_result.get("location", {}).get("country", "Unknown"),
                    "city": analysis_result.get("location", {}).get("city", "Unknown"),
                    "neighborhood": analysis_result.get("location", {}).get("neighborhood", "Unknown"),
                    "street": analysis_result.get("location", {}).get("street", "Unknown"),
                    "coordinates": {
                        "latitude": float(analysis_result.get("location", {}).get("coordinates", {}).get("latitude", "0").replace("°", "").replace("S", "-").replace("N", "").strip()),
                        "longitude": float(analysis_result.get("location", {}).get("coordinates", {}).get("longitude", "0").replace("°", "").replace("W", "-").replace("E", "").strip())
                    },
                    "architectural_features": analysis_result.get("architectural_features", []),
                    "landscape_features": analysis_result.get("landscape_features", [])
                }
            }

        except Exception as e:
            logger.error(f"Error in analyze_image: {str(e)}")
            return {
                "llm_analysis": {
                    "error": str(e)
                },
                "geo_data": {
                    "error": str(e)
                }
            }

    def chat_about_image(self, image_path: str, user_message: str) -> Dict[str, Any]:
        """Chat about an image using Gemini Vision."""
        try:
            # Load and prepare the image
            image = Image.open(image_path)
            
            # Prepare the prompt
            prompt = f"""Analiza esta imagen y responde a la siguiente pregunta: {user_message}
            
            IMPORTANTE: 
            - Enfócate SOLO en elementos arquitectónicos y geográficos
            - NO incluyas análisis de personas
            - Proporciona coordenadas aproximadas basadas en características visibles
            - Indica el nivel de confianza en la geolocalización"""

            # Generate response
            response = self.model.generate_content([prompt, image])
            
            return {
                "llm_analysis": {
                    "response": response.text
                }
            }

        except Exception as e:
            logger.error(f"Error in chat_about_image: {str(e)}")
            return {
                "llm_analysis": {
                    "error": str(e)
                }
            }
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
    
    def extract_location_from_frame(self, video_frame_path: str) -> Dict[str, Any]:
        """
        Extract location information from a video frame.
        
        Args:
            video_frame_path: Path to the video frame image
            
        Returns:
            Dictionary with location information
        """
        # Ensure we have bytes
        with open(video_frame_path, "rb") as f:
            image_data = f.read()
            
        return self.analyze_image(video_frame_path)
    
    def rate_limit_check(self):
        """
        Implement rate limiting to avoid API quota issues.
        Waits for a short period to avoid exceeding rate limits.
        """
        time.sleep(1)  # Simple rate limiting strategy 