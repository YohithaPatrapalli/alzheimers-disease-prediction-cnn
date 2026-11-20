"""
URL Configuration for alzheimers_prediction project.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse

# Simple home view
def home_view(request):
    return HttpResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alzheimer's Disease Prediction System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 50px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 20px;
            }
            a {
                color: white;
                text-decoration: none;
                background: rgba(255,255,255,0.2);
                padding: 10px 20px;
                border-radius: 10px;
                display: inline-block;
                margin: 10px;
            }
            a:hover {
                background: rgba(255,255,255,0.3);
            }
            .btn {
                display: inline-block;
                margin: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Alzheimer's Disease Prediction System</h1>
            <p>AI-powered early detection using Deep Learning CNN models</p>
            <p>Multi-stage classification: Non-Demented, Very Mild, Mild, Moderate</p>
            <div>
                <a href="/admin/" class="btn">Admin Panel</a>
                <a href="/prediction/static/" class="btn">Static Prediction</a>
                <a href="/prediction/dynamic/" class="btn">Dynamic Prediction (QR Code)</a>
                <a href="/prediction/models/" class="btn">View Models</a>
                <a href="/dataset/" class="btn">Datasets</a>
            </div>
            <p style="margin-top: 30px; font-size: 12px;">System is running!</p>
        </div>
    </body>
    </html>
    """)

# Define URL patterns - INCLUDE ALL APP URLs
from django.contrib.auth import views as auth_views
urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # Home page
    path('', home_view, name='home'),
    
    # Include core app URLs
    path('', include(('core.urls', 'core'), namespace='core')),
    
    # Include dataset_manager app URLs
    path('dataset/', include('dataset_manager.urls')),
    
    # Include prediction app URLs  
    path('prediction/', include('prediction.urls')),

    path('accounts/', include('django.contrib.auth.urls')),
]

# Serve static and media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
else:
    from core.views import private_media

    urlpatterns += [
        path('media/<path:path>', private_media, name='private_media'),
    ]