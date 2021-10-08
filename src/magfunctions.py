import os
from pathlib import Path


def create_export_path(path: str) -> str:
    """create an export path and return the path string"""
    if os.path.exists(path) == False:
        os.mkdir(path)

    return path
