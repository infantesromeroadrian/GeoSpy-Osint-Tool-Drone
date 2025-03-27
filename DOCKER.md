# Docker Deployment Guide for Drone-OSINT-GeoSpy

This guide explains how to deploy the Drone-OSINT-GeoSpy application using Docker containers.

## Prerequisites

- Docker Engine installed (version 19.03.0+)
- Docker Compose installed (version 1.27.0+)
- Git (for cloning the repository)

## Development Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/Drone-Osint-GeoSpy.git
cd Drone-Osint-GeoSpy
```

### 2. Set Up Environment Variables

Check your `.env` file to ensure it contains your API keys:

```bash
# Contents should include:
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

If you don't have an `.env` file, create one with these necessary keys.

### 3. Build and Start the Services

```bash
docker-compose up -d
```

This will:
- Build the Docker image
- Start both the backend and frontend services
- Create a shared network for the containers to communicate
- Mount the `./data` directory as a volume for persistent storage

### 4. Accessing the Application

- Frontend (Streamlit): http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### 5. Stopping the Services

```bash
docker-compose down
```

Add the `-v` flag to also remove the volumes: `docker-compose down -v`

## Production Deployment

For production environments, we provide an enhanced configuration with:
- Nginx as a reverse proxy
- Volume management for persistent data
- Resource limits
- Improved logging
- Optional HTTPS configuration

### 1. Prepare the Production Environment

```bash
cp .env.prod.template .env.prod
```

Edit `.env.prod` to add your API keys and adjust any configuration settings.

### 2. Build and Start the Production Services

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Enabling HTTPS (Optional)

1. Generate or obtain SSL certificates
2. Place them in a directory called `ssl/`:
   ```bash
   mkdir -p ssl
   cp /path/to/your/cert.pem ssl/
   cp /path/to/your/key.pem ssl/
   ```
3. Uncomment the SSL-related lines in `docker-compose.prod.yml`
4. Modify `nginx.conf` to enable HTTPS

### 4. Scaling the Services

You can scale the backend service for better performance:

```bash
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

Note: When scaling the backend, you'll need to ensure the frontend connects to the correct backend instance.

## Container Structure

- **Backend Container**: Runs the FastAPI backend server
- **Frontend Container**: Runs the Streamlit frontend server
- **Nginx Container**: (Production only) Acts as a reverse proxy

## Data Persistence

All data is stored in the `data/` directory, which is mounted as a volume in the containers. This includes:
- Uploaded images in `data/uploads/`
- Extracted frames in `data/frames/`
- Analysis results in `data/results/`

## Troubleshooting

### Common Issues

1. **Error: Address already in use**
   - You have a service already running on port 8000 or 8501
   - Solution: Change the port mapping in docker-compose.yml

2. **Container exits immediately**
   - Check container logs: `docker-compose logs backend` or `docker-compose logs frontend`
   - Ensure your .env file is properly configured

3. **API Key errors**
   - Ensure your OpenAI API key is correctly set in the .env file
   - Verify that the API key has GPT-4V access

4. **Volume permission issues**
   - If you encounter permission errors: `sudo chown -R $(id -u):$(id -g) ./data`

## Maintenance

### Updating the Application

1. Pull the latest code:
   ```bash
   git pull
   ```

2. Rebuild and restart the containers:
   ```bash
   docker-compose up -d --build
   ```

### Viewing Logs

```bash
# All logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Specific service logs
docker-compose logs backend
docker-compose logs frontend
```

### Backup and Restore

To backup the data:
```bash
tar -czvf backup.tar.gz ./data
```

To restore from a backup:
```bash
tar -xzvf backup.tar.gz -C /path/to/restore
``` 