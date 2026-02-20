import os

def get_absolute_path(relative_path):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, relative_path)

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)