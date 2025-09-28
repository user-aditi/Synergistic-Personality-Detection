import uvicorn
from config import API_CONFIG

def run_server():
    """Run the FastAPI server"""
    print("Starting OCEAN Personality Analyzer API...")
    print(f"Server will run at: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print(f"API documentation: http://{API_CONFIG['host']}:{API_CONFIG['port']}/docs")
    print("Press Ctrl+C to stop the server")

    # --- NEW DIAGNOSTIC PRINTS ---
    print("\n[DEBUG] Attempting to start Uvicorn server...")

    uvicorn.run(
        "src.api:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=False, 
        log_level=API_CONFIG['log_level']
    )

    print("[DEBUG] This line should never be printed if the server runs correctly.")
    
if __name__ == "__main__":
    run_server()
