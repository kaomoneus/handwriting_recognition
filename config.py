"""
Default cache directory value
"""
import os
from pathlib import Path

CACHE_DIR_DEFAULT = Path(os.environ["HOME"]) / ".handwritten_preprocessing_cache"
