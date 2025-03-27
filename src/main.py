import os
import uvicorn
from dotenv import load_dotenv
import sys
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

# Intentar configurar el archivo .env si es necesario
try:
    from src.utils.env_setup import setup_env_file
    setup_env_file()
except ImportError:
    # Si estamos en Docker, la ruta de importación es diferente
    try:
        from utils.env_setup import setup_env_file
        setup_env_file()
    except Exception as e:
        logger.warning(f"Could not setup environment file: {e}")

# Load environment variables
load_dotenv()

def main():
    """
    Main entry point for the Drone OSINT GeoSpy application.
    """
    # Log environment details for debugging
    logger.info(f"Starting Drone OSINT GeoSpy server")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    logger.info(f"OpenAI API Key present: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
    
    # Ensure required directories exist
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/frames", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    logger.info("Data directories created/verified")
    
    # Determine if running in Docker or not
    in_docker = "/app" in os.environ.get("PYTHONPATH", "") or os.environ.get("DOCKER_CONTAINER") == "1"
    
    if in_docker:
        # Docker environment
        module_path = "backend.api:app"
        logger.info("Running in Docker environment")
    else:
        # Local environment
        module_path = "src.backend.api:app"
        logger.info("Running in local environment")
    
    # Print for debugging
    logger.info(f"Starting server with module path: {module_path}")
    
    # Start the FastAPI server
    logger.info("Launching uvicorn server...")
    uvicorn.run(
        module_path,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1) 