import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import system_router, models_router, generate_router
from .manager import manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: auto-load models if environment variables are set
    manager.initialize_from_env()
    yield
    # Shutdown cleanup
    for p in manager.pipelines:
        p.clear()


app = FastAPI(
    title="xfusion Inference API",
    description="High-performance decoupled backend for diffusion inference, designed for CDN/Web clients.",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS for CDN and frontend clients
cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(system_router)
app.include_router(models_router)
app.include_router(generate_router)


@app.get("/")
async def root():
    return {
        "name": "xfusion Inference API",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/api/v1/system/health"
    }
