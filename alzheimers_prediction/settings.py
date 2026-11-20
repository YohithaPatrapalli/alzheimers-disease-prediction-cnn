"""
Django settings for alzheimers_prediction project.
Generated for Alzheimer's Disease Prediction System
"""

import os
from pathlib import Path
from decouple import config
from django.core.exceptions import ImproperlyConfigured
import sys

# Build paths inside the project like this: BASE_DIR / 'subdir'
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = config('SECRET_KEY', default='django-insecure-8x!q3#p@k9m2n5b7v9c1x3z5a7s8d9f0g1h2j3k4l5')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True, cast=bool)

if not DEBUG and SECRET_KEY.startswith('django-insecure-'):
    raise ImproperlyConfigured('Set a production SECRET_KEY when DEBUG is False.')

ALLOWED_HOSTS = [
    host.strip()
    for host in config('ALLOWED_HOSTS', default='localhost,127.0.0.1').split(',')
    if host.strip()
]
RENDER_EXTERNAL_HOSTNAME = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
if RENDER_EXTERNAL_HOSTNAME and RENDER_EXTERNAL_HOSTNAME not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)

CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in config('CSRF_TRUSTED_ORIGINS', default='').split(',')
    if origin.strip()
]
if RENDER_EXTERNAL_HOSTNAME:
    render_origin = f'https://{RENDER_EXTERNAL_HOSTNAME}'
    if render_origin not in CSRF_TRUSTED_ORIGINS:
        CSRF_TRUSTED_ORIGINS.append(render_origin)

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party apps
    'crispy_forms',
    'crispy_bootstrap5',
    
    # Local apps
    'core',
    'dataset_manager',
    'prediction',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'alzheimers_prediction.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'core.context_processors.site_settings',
                'core.context_processors.user_notifications',
            ],
        },
    },
]

WSGI_APPLICATION = 'alzheimers_prediction.wsgi.application'

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': config('DATABASE_PATH', default=str(BASE_DIR / 'db.sqlite3')),
    }
}

# Use PostgreSQL in production (optional)
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql',
#         'NAME': config('DB_NAME', 'alzheimers_db'),
#         'USER': config('DB_USER', 'postgres'),
#         'PASSWORD': config('DB_PASSWORD', ''),
#         'HOST': config('DB_HOST', 'localhost'),
#         'PORT': config('DB_PORT', '5432'),
#     }
# }

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'
if not DEBUG:
    STATICFILES_STORAGE = 'whitenoise.storage.CompressedStaticFilesStorage'

# Media files (User uploaded content)
MEDIA_URL = '/media/'
MEDIA_ROOT = Path(config('MEDIA_ROOT', default=str(BASE_DIR / 'media')))

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Crispy Forms Configuration
CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"

# Authentication URLs
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'core:dashboard'
LOGOUT_REDIRECT_URL = 'index'

# Email Configuration (for development)
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
EMAIL_HOST = config('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = config('EMAIL_PORT', 587, cast=int)
EMAIL_USE_TLS = config('EMAIL_USE_TLS', True, cast=bool)
EMAIL_HOST_USER = config('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD', '')
DEFAULT_FROM_EMAIL = config('DEFAULT_FROM_EMAIL', 'noreply@alzheimersprediction.com')
CONTACT_EMAIL = config('CONTACT_EMAIL', 'contact@alzheimersprediction.com')

# Session Configuration
SESSION_COOKIE_AGE = 86400  # 24 hours in seconds
SESSION_SAVE_EVERY_REQUEST = True

# File Upload Settings
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB

# ML Model Settings
# ============================================================================
# Machine Learning configuration for Alzheimer's Disease Prediction

# Model paths
MODEL_PATH = Path(config(
    'MODEL_DIRECTORY',
    default=str(BASE_DIR / 'ml_models' / 'saved_models'),
))
CHECKPOINT_PATH = Path(config(
    'CHECKPOINT_DIRECTORY',
    default=str(BASE_DIR / 'ml_models' / 'checkpoints'),
))

# Model parameters
DEFAULT_IMAGE_SIZE = (224, 224)  # Input image size for CNN
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Class names for Alzheimer's stages
CLASS_NAMES = [
    'Non-Demented',
    'Very Mild Demented',
    'Mild Demented',
    'Moderate Demented'
]

# Stage mapping for database
STAGE_MAPPING = {
    'non_demented': 'Non-Demented',
    'very_mild': 'Very Mild Demented',
    'mild': 'Mild Demented',
    'moderate': 'Moderate Demented'
}

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence for valid prediction
HIGH_CONFIDENCE = 0.85  # High confidence threshold
LOW_CONFIDENCE = 0.70  # Low confidence threshold

# Data Augmentation Settings
AUGMENTATION_SETTINGS = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest',
    'brightness_range': [0.8, 1.2]
}

# Available model architectures
AVAILABLE_MODELS = [
    ('cnn', 'Custom CNN'),
    ('vgg16', 'VGG16'),
    ('resnet50', 'ResNet50'),
    ('inceptionv3', 'InceptionV3'),
    ('densenet121', 'DenseNet121'),
    ('mobilenetv2', 'MobileNetV2'),
    ('efficientnetb0', 'EfficientNetB0')
]

# Training settings
TRAINING_SETTINGS = {
    'early_stopping_patience': 15,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.2,
    'min_lr': 1e-7,
    'validation_split': 0.2,
    'test_split': 0.15
}

# Dataset Settings
DATASET_SETTINGS = {
    'max_dataset_size': 10 * 1024 * 1024 * 1024,  # 10GB
    'allowed_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'min_image_size': (100, 100),
    'max_image_size': (5000, 5000),
    'supported_archives': ['.zip', '.rar', '.7z']
}

# QR Code Settings
QR_CODE_SETTINGS = {
    'version': 1,
    'box_size': 10,
    'border': 4,
    'fill_color': 'black',
    'back_color': 'white'
}

# Cache Settings (optional)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}

# Use Redis for production (optional)
# CACHES = {
#     'default': {
#         'BACKEND': 'django_redis.cache.RedisCache',
#         'LOCATION': config('REDIS_URL', 'redis://127.0.0.1:6379/1'),
#         'OPTIONS': {
#             'CLIENT_CLASS': 'django_redis.client.DefaultClient',
#         }
#     }
# }

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'stream': sys.stdout,
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'django.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'ml_models': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'prediction': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# Create logs directory if it doesn't exist
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# Security Settings (for production)
if not DEBUG:
    # HTTPS settings
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    
    # Security headers
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    
    # Session security
    SESSION_COOKIE_HTTPONLY = True
    CSRF_COOKIE_HTTPONLY = True
    
    # Password validation
    AUTH_PASSWORD_VALIDATORS = [
        {
            'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
        },
        {
            'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
            'OPTIONS': {
                'min_length': 12,
            }
        },
        {
            'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
        },
        {
            'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
        },
    ]

# Performance Settings
DATA_UPLOAD_MAX_NUMBER_FIELDS = 10000  # Increase if needed
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB

# Celery Configuration (optional - for async tasks)
# CELERY_BROKER_URL = config('REDIS_URL', 'redis://localhost:6379')
# CELERY_RESULT_BACKEND = 'django-db'
# CELERY_ACCEPT_CONTENT = ['application/json']
# CELERY_TASK_SERIALIZER = 'json'
# CELERY_RESULT_SERIALIZER = 'json'
# CELERY_TIMEZONE = TIME_ZONE

# API Settings (if using REST API)
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/day',
        'user': '1000/day'
    }
}

# Site Settings
SITE_NAME = "Alzheimer's Disease Prediction System"
SITE_DESCRIPTION = "AI-powered early detection and multi-stage classification of Alzheimer's disease using deep learning CNN models"
SITE_VERSION = "1.0.0"

# Admin Site Configuration
ADMIN_SITE_HEADER = "Alzheimer's Disease Prediction Administration"
ADMIN_SITE_TITLE = "AD Prediction Admin"
ADMIN_INDEX_TITLE = "Welcome to Alzheimer's Disease Prediction System Administration"

# Create necessary directories
for directory in [MEDIA_ROOT, STATIC_ROOT, MODEL_PATH, CHECKPOINT_PATH]:
    directory.mkdir(parents=True, exist_ok=True)

# Create media subdirectories
media_subdirs = ['datasets', 'models', 'predictions', 'qrcodes', 'avatars', 'temp']
for subdir in media_subdirs:
    (MEDIA_ROOT / subdir).mkdir(parents=True, exist_ok=True)

print(f"Settings loaded. DEBUG={DEBUG}, MEDIA_ROOT={MEDIA_ROOT}, STATIC_ROOT={STATIC_ROOT}")