"""
ASGI config for alzheimers_prediction project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os
import sys
from pathlib import Path
from django.core.asgi import get_asgi_application

# Add the project directory to the Python path
project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir))

# Set the default settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'alzheimers_prediction.settings')

# Get the ASGI application
application = get_asgi_application()