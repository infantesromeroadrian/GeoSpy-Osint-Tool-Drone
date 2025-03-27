import os
import streamlit.web.cli as stcli
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_streamlit():
    """Run the Streamlit frontend application."""
    sys.argv = [
        "streamlit",
        "run",
        "src/frontend/app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    
    # Create needed directories
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/frames", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    
    sys.exit(stcli.main())

if __name__ == "__main__":
    run_streamlit() 