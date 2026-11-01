import os

_ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("PYTORCH_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import uvicorn


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("DEBUG", "false").lower() in ["true", "1"]

    print(f"Starting xfusion API Server on {host}:{port} ...")
    uvicorn.run("server.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()

