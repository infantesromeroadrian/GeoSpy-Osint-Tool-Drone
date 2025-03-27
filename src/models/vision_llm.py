import os
import json
import base64
from typing import Dict, Any, List, Optional, Union
import openai
from dotenv import load_dotenv
import time
import datetime
import sys
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class VisionLLM:
    """
    Class to interact with Vision-capable LLMs (like OpenAI's GPT-4V) to analyze images
    and extract geolocation and terrain information.
    """
    
    def __init__(self, api_key: str):
        """Initialize the VisionLLM with OpenAI API key"""
        # Establece el cliente como None por defecto
        self.client = None
        self.conversation_history = []
        
        # Verificar API key
        if not api_key:
            logger.warning("API key is empty or None")
            return
            
        key_length = len(api_key)
        key_preview = f"{api_key[:4]}...{api_key[-4:]}" if key_length > 8 else "Invalid key format"
        logger.info(f"Initializing VisionLLM with API key of length {key_length}: {key_preview}")
        
        # Configurar API key para OpenAI
        try:
            # Usar cliente OpenAI a través de importación directa
            from openai import OpenAI
            
            # Intentar método estándar primero
            try:
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully with standard method")
            except TypeError as e:
                if "proxies" in str(e):
                    logger.warning(f"Error with 'proxies' parameter, trying alternative method: {e}")
                    # Método alternativo: crear y luego asignar api_key
                    self.client = OpenAI()
                    self.client.api_key = api_key
                    logger.info("OpenAI client initialized successfully with alternative method")
                else:
                    raise
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
    
    def _encode_image(self, image_data: bytes) -> str:
        """Encode image data to base64"""
        if isinstance(image_data, str):
            # Si recibimos una ruta de archivo en lugar de datos binarios
            try:
                with open(image_data, "rb") as f:
                    image_data = f.read()
                logger.info(f"Loaded image data from file: {image_data}")
            except Exception as e:
                logger.error(f"Error loading image file: {str(e)}")
                raise ValueError(f"Invalid image path: {image_data}")
                
        return base64.b64encode(image_data).decode('utf-8')
    
    def analyze_image(self, image_data: bytes, image_name: str = None, prompt_type: str = "general") -> Dict:
        """
        Analyze an image using OpenAI Vision model
        
        Args:
            image_data: The binary data of the image
            image_name: Optional name of the image for reference
            prompt_type: Type of analysis to perform (general, geo, terrain)
            
        Returns:
            Dictionary with analysis results
        """
        if not self.client:
            logger.error("OpenAI client not initialized")
            return {"error": "OpenAI client not initialized"}
        
        try:
            # Check rate limits
            self.rate_limit_check()
            
            # Encode image (asegurando que es bytes)
            if isinstance(image_data, str):
                with open(image_data, "rb") as f:
                    image_data = f.read()
                    
            base64_image = self._encode_image(image_data)
            
            # Select prompt based on type
            if prompt_type == "geo":
                system_prompt = """Eres un analista especializado en geolocalización de imágenes. Tu tarea es analizar ÚNICAMENTE los elementos arquitectónicos, paisajísticos y geográficos de la imagen para determinar su ubicación probable.

Por favor, analiza la siguiente imagen y proporciona:

1. Descripción detallada del entorno físico:
   - Edificios y arquitectura (estilo, altura, características distintivas)
   - Monumentos o estructuras notables
   - Señales o textos visibles (idioma, nombres de calles)
   - Características geográficas (montañas, ríos, costa, etc.)
   - Vegetación y paisaje
   - Condiciones ambientales y clima visible
   - Infraestructura urbana (diseño de calles, transportes)

2. Proporciona una tabla con posibles ubicaciones, usando exactamente este formato:
| Nivel de confianza | País | Ciudad/Región | Barrio/Distrito | Lugar específico | Coordenadas aproximadas |
|-------------------|------|---------------|-----------------|------------------------|-------------------------|
| 85% | España | Madrid | Centro | Calle Gran Vía | 40.4200, -3.7025 |
| 65% | Francia | París | Le Marais | Rue de Rivoli | 48.8566, 2.3522 |

3. Para la ubicación más probable:
   - Puntos de referencia cercanos
   - Dirección exacta si es posible
   - Orientación de la cámara

4. Explica tu razonamiento:
   - Elementos que sustentan tu análisis
   - Razones para los niveles de confianza asignados

Importante: NO incluyas NINGUNA mención, análisis o descripción de personas en tu respuesta. Enfócate EXCLUSIVAMENTE en elementos geográficos, arquitectónicos y de infraestructura."""
            elif prompt_type == "terrain":
                system_prompt = """You are a terrain and environment analysis expert.
                Analyze this drone image and provide detailed information about:
                1. Terrain type (mountainous, flat, urban, rural, coastal, etc.)
                2. Vegetation and ecosystem details
                3. Weather conditions and visible climate indicators
                4. Geological features and formations
                5. Land use patterns visible in the image
                6. Any environmental concerns or notable ecological features
                Be technical and precise. Focus only on the physical and environmental aspects visible in the image."""
            else:  # general
                system_prompt = """Eres un analista de imágenes profesional.
                Examina esta imagen y proporciona:
                1. Una descripción detallada del entorno físico y elementos arquitectónicos visibles
                2. Elementos notables de paisaje, infraestructura y características del terreno
                3. Condiciones ambientales y meteorológicas observables
                4. Patrones o características inusuales de interés
                5. Propósito probable de esta toma
                
                Importante: NO incluyas ninguna mención, análisis o descripción de personas en tu respuesta.
                Sé objetivo y minucioso en tu evaluación. Describe solo lo que puedes ver con confianza."""
                
            # Create the message content
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Por favor analiza SOLO los elementos arquitectónicos y geográficos de esta imagen para determinar su ubicación. NO ANALICES personas o individuos."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Use el cliente OpenAI
            try:
                # Make the API call with client
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=1000
                )
                
                # Get the response content
                analysis = response.choices[0].message.content
                logger.info(f"Successfully received analysis from OpenAI")
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {str(e)}")
                return {"error": f"Failed to call OpenAI API: {str(e)}"}
            
            # Record to conversation history
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.conversation_history.append({
                "timestamp": timestamp,
                "type": prompt_type,
                "image_name": image_name,
                "response": analysis
            })
            
            # Return the analysis
            return {
                "timestamp": timestamp,
                "analysis": analysis,
                "prompt_type": prompt_type,
                "image_name": image_name,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_image: {str(e)}")
            return {"error": str(e)}
    
    def chat_about_image(self, image_path: str, user_message: str) -> str:
        """
        Have a conversation with the Vision LLM about an image.
        
        Args:
            image_path: Path to the image file
            user_message: User's message/question about the image
            
        Returns:
            LLM's response
        """
        try:
            # Make sure we have bytes
            with open(image_path, "rb") as f:
                image_data = f.read()
                
            # Encode image
            base64_image = self._encode_image(image_data)
            
            # Determinar si la consulta es sobre geolocalización
            geo_keywords = ["ubicación", "geo", "donde", "país", "ciudad", "coordenadas", "lugar", "calle", 
                            "location", "where", "country", "city", "coordinates", "place", "street"]
            
            is_geo_query = any(keyword in user_message.lower() for keyword in geo_keywords)
            
            # Build the conversation history with appropriate system prompt
            if is_geo_query:
                # Usar el prompt de geolocalización si la consulta es sobre ubicación
                system_prompt = """Eres un analista especializado en geolocalización de imágenes. Tu tarea es analizar ÚNICAMENTE los elementos arquitectónicos, paisajísticos y geográficos de la imagen para determinar su ubicación probable.

Proporciona información sobre:
- País, ciudad, barrio y calle visible en la imagen
- Coordenadas geográficas aproximadas
- Puntos de referencia y características arquitectónicas distintivas

Presenta la información en formato claro y conciso.

IMPORTANTE: NO incluyas NINGUNA mención, análisis o descripción de personas en tu respuesta."""
            else:
                # Prompt genérico para otras consultas
                system_prompt = """Eres un analista de imágenes profesional especializado en características geográficas, arquitectónicas y ambientales. Analiza las imágenes proporcionadas y responde a las preguntas sobre estos elementos con precisión y detalle.

IMPORTANTE: NO incluyas NINGUNA mención, análisis o descripción de personas en tu respuesta."""
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            
            # Add conversation history if it exists
            for msg in self.conversation_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
            
            # Add the current user message with the image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{user_message} (Analiza SOLO elementos arquitectónicos y de paisaje. NO incluyas análisis de personas)"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            })
            
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.5
            )
            
            # Extract the response
            result = response.choices[0].message.content
            
            # Add the response to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": result})
            
            return result
            
        except Exception as e:
            logger.error(f"Error in chat_about_image: {str(e)}")
            return f"Error communicating with Vision LLM: {str(e)}"
    
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
            
        return self.analyze_image(image_data, video_frame_path, "geo")
    
    def rate_limit_check(self):
        """
        Implement rate limiting to avoid API quota issues.
        Waits for a short period to avoid exceeding rate limits.
        """
        time.sleep(1)  # Simple rate limiting strategy 