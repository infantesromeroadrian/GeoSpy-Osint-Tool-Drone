# Drone-OSINT-GeoSpy

An advanced military-grade drone surveillance application that analyzes terrain imagery in real-time using Vision LLMs. The system can extract exact geolocation data, analyze terrain features, and provide an interactive interface for querying the vision model about details in the imagery.

## Features

- **Real-time and Static Image Analysis**: Process both live drone video feeds and static images.
- **Precise Geolocation**: Identify country, city, district, neighborhood, street, and coordinates.
- **Metadata Extraction**: Extract geolocation data from image metadata.
- **Interactive LLM Analysis**: Chat-based interface to query the vision model about specific details in the imagery.
- **Military-style UI**: Tactical interface designed for operational use.

## System Architecture

The system consists of several integrated modules:
- **Image Processing**: Handles image capture, preprocessing, and metadata extraction.
- **Vision LLM Integration**: Connects to advanced vision models for detailed image analysis.
- **Geolocation Services**: Processes and verifies location data from multiple sources.
- **Interactive Interface**: Military-style UI for real-time interaction with the system.

## Setup and Installation

### Prerequisites

- Python 3.9+
- OpenAI API key (with GPT-4V access)
- Google Maps API key (optional, for enhanced geolocation)

### Option 1: Standard Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-org/Drone-Osint-GeoSpy.git
   cd Drone-Osint-GeoSpy
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. Configure your API keys in the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
   ```

### Option 2: Docker Installation (Recommended)

1. Clone the repository:
   ```
   git clone https://github.com/your-org/Drone-Osint-GeoSpy.git
   cd Drone-Osint-GeoSpy
   ```

2. Check your environment file:
   - For development: Copy `.env.example` to `.env` and add your API keys
   - For production: `cp .env.example .env.prod` and edit to add your API keys

3. Build and start Docker containers:
   ```
   docker-compose up -d
   ```
   
   For more detailed Docker instructions, see [DOCKER.md](DOCKER.md).

### Option 3: Using the run.py Helper Script

For simplified setup and running, use the provided `run.py` script:

```bash
# Create and configure .env file from template if needed
python run.py --mode docker     # Run with Docker (default)
python run.py --mode local      # Run locally without Docker
python run.py --mode restart    # Restart Docker containers
python run.py --mode production # Run with production Docker configuration
python run.py --mode frontend   # Run only the frontend (local mode)
```

## Running the Application

### Option 1: Standard Mode

1. Start the FastAPI backend server:
   ```
   python src/main.py
   ```
   This will start the backend server at http://localhost:8000

2. In a separate terminal, start the Streamlit frontend:
   ```
   python -m streamlit run src/frontend/app.py
   ```
   This will start the frontend at http://localhost:8501

### Option 2: Docker Mode

The application will be automatically started when you run:
```
docker-compose up -d
```

Access the application at:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Using the Application

1. **Image Analysis Tab**:
   - Upload drone imagery
   - View detailed geolocation analysis
   - Examine confidence levels and evidence

2. **Drone Stream Tab**:
   - Connect to live drone feeds via RTSP/HTTP URLs
   - Capture and analyze frames in real-time
   - Monitor changing terrain and locations

3. **Interrogation Tab**:
   - Chat with the Vision LLM about specific details in the imagery
   - Ask about terrain features, structures, and potential hazards
   - Get tactical insights about the observed location

## Testing

To test the application with sample data:

1. Add test images to the `data/test_images` directory
2. Run the test suite:
   ```
   python -m pytest tests/
   ```

## Troubleshooting

If you encounter issues with the application, try the following steps:

### 1. Run the Troubleshooting Script

We've included a troubleshooting script that automatically detects common issues:

```bash
# Make the script executable first if needed
chmod +x troubleshoot.sh
./troubleshoot.sh
```

### 2. Common Issues and Solutions:

- **API Key Issues**: 
  - Ensure your `.env` file contains a valid OpenAI API key
  - The key should start with "sk-" and should have access to GPT-4 Vision models

- **Docker Connection Issues**:
  - Reset the Docker network with `docker network prune`
  - Rebuild all containers with `docker-compose down --rmi all && docker-compose up --build`
  - Check container logs with `docker-compose logs backend` or `docker-compose logs frontend`

- **Backend Initialization Issues**:
  - Use our debug tool: `docker-compose run --rm backend debug`
  - Check backend module initialization: `docker-compose run --rm backend test-backend`
  - Ensure the OpenAI API version is compatible (current code uses v1.6.1)

- **Frontend Can't Connect to Backend**:
  - Ensure the API_URL environment variable is set to http://backend:8000 in Docker
  - Check if the backend container is healthy with `docker-compose ps`
  - Test the connection: `docker-compose run --rm frontend python src/frontend/api_test.py`

### 3. Debugging Mode

You can run the application with additional debugging information:

```bash
# Set debug level in .env
DEBUG=true
LOG_LEVEL=DEBUG

# Run with debug enabled
docker-compose up -d
docker-compose logs -f
```

For more detailed Docker troubleshooting, see [DOCKER.md](DOCKER.md).

## Security Notes

- This application processes potentially sensitive geolocation data
- API keys should be kept secure and never committed to version control
- Consider network isolation when deploying in sensitive environments
- For production deployment, use the `docker-compose.prod.yml` configuration

## License

This project is licensed under the MIT License - see the LICENSE file for details. 