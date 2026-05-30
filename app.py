import os
import uvicorn
from dotenv import load_dotenv

# Load variables
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
    
    print(f"[*] Starting AETHER Facet Scoring Engine at http://{host}:{port}")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=debug)
