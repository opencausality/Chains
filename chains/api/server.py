"""FastAPI server for Chains."""

from __future__ import annotations

from fastapi import FastAPI

from chains import __version__
from chains.api.routes import router


def create_app() -> FastAPI:
    application = FastAPI(title="Chains API", version=__version__,
                          description="Debug multi-step LLM pipelines causally.")
    application.include_router(router)
    return application


app = create_app()
