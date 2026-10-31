"""Run with: uvicorn main:app --reload"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from app import app

__all__ = ["app"]
