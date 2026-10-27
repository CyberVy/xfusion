import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("DEBUG", "false").lower() in ["true", "1"]

    print(f"Starting xfusion API Server on {host}:{port} ...")
    uvicorn.run("server.api.server:app", host=host, port=port, reload=reload)
