import os
import sys

# Add the project root to sys.path so modules like api_server can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import app
