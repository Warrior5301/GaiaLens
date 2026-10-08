"""
EarthDial FastAPI Service (Facade)
==================================

Maintains backward compatibility by re-exporting `app` from `app.py`.
"""

from app import app

__all__ = ["app"]
