"""
Routers package - exports all API routers for the application.
"""

from . import agents_router
from . import automate_router
from . import chat_router
from . import chunking_router
from . import config_router
from . import dataset_router
from . import document_router
from . import evaluation_router
from . import health_router
from . import huggingface_router
from . import model_router
from . import processed_dataset_router
from . import raw_dataset_router
from . import template_router
from . import upload_router

__all__ = [
    "agents_router",
    "automate_router",
    "chat_router",
    "chunking_router",
    "config_router",
    "dataset_router",
    "document_router",
    "evaluation_router",
    "health_router",
    "huggingface_router",
    "model_router",
    "processed_dataset_router",
    "raw_dataset_router",
    "template_router",
    "upload_router",
]
