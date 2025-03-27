import requests
import os
import time
import sys

def test_backend_connection():
    """Test the connection to the backend API"""
    # Get API URL from environment or use default
    api_url = os.environ.get("API_URL", "http://backend:8000")
    print(f"Testing connection to backend at: {api_url}")
    
    # Try to connect to the health endpoint
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Attempt {attempt}/{max_attempts}...")
            response = requests.get(f"{api_url}/api/session/health", timeout=5)
            if response.status_code == 200:
                print(f"SUCCESS! Backend is reachable. Response: {response.json()}")
                return True
            else:
                print(f"Got response with status code {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
        
        # Wait before retry
        if attempt < max_attempts:
            print("Waiting 2 seconds before next attempt...")
            time.sleep(2)
    
    print("FAILED to connect to backend after multiple attempts")
    return False

if __name__ == "__main__":
    print("API Connection Test Utility")
    print("==========================")
    print(f"Environment variables:")
    print(f"API_URL: {os.environ.get('API_URL', 'Not set')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"Current directory: {os.getcwd()}")
    print("==========================")
    
    # Run the test
    success = test_backend_connection()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 