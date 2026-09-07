"""Convenience entry point: `python scripts/run_server.py`."""
import os
import uvicorn
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    port = int(os.environ.get("API_PORT", 8001))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=True)
