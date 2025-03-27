import os
import sys
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('env_setup')

def setup_env_file():
    """
    Copy the .env.example file to .env if .env doesn't exist.
    This helps users get started quickly with the correct environment variables.
    """
    # Determine the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we're in Docker
    if '/app' in os.environ.get('PYTHONPATH', ''):
        project_root = '/app'
    else:
        # In local development, go up two directories from the utils folder
        project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Define file paths
    env_example_file = os.path.join(project_root, '.env.example')
    env_file = os.path.join(project_root, '.env')
    
    # Check if .env file already exists
    if os.path.exists(env_file):
        logger.info(".env file already exists. Skipping creation.")
        return
    
    # Check if .env.example exists
    if not os.path.exists(env_example_file):
        logger.error(".env.example file not found! Cannot create .env file.")
        return
    
    # Copy .env.example to .env
    try:
        shutil.copy2(env_example_file, env_file)
        logger.info(f".env file created from .env.example. Please update it with your API keys.")
        logger.info(f"File location: {env_file}")
    except Exception as e:
        logger.error(f"Failed to create .env file: {e}")

if __name__ == "__main__":
    setup_env_file()
    print("Environment setup complete!") 