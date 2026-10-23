from .system import router as system_router
from .models import router as models_router
from .generate import router as generate_router

__all__ = ["system_router", "models_router", "generate_router"]
