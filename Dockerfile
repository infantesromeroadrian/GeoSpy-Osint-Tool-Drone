FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DOCKER_CONTAINER=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for OpenEXIF and metadata extraction
    libexif-dev \
    libglib2.0-0 \
    # Required for OpenCV
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Required for YOLOv8 and performance
    ffmpeg \
    libavcodec-extra \
    # Required for healthchecks
    curl \
    # Cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Create and set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for YOLOv8 models and download the default model
RUN mkdir -p /app/models/yolo \
    && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy project files
COPY . /app/

# Make sure the entrypoint script is executable
RUN chmod +x /app/docker-entrypoint.sh

# Set up volume for data persistence
VOLUME ["/app/data"]

# Expose ports - Both backend and frontend
EXPOSE 8000 8501

# Set entrypoint 
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["all"] 