import sys
import os

# Automatically append the project root directory to sys.path so tests can import local modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
