import random
import string
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
import os
import uuid
import json
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def generate_random_string(length=32):
    """Generate random string of specified length"""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def generate_verification_token():
    """Generate email verification token"""
    return str(uuid.uuid4()) + hashlib.sha256(
        str(random.getrandbits(256)).encode()
    ).hexdigest()

def send_verification_email(user, request):
    """Send email verification link to user"""
    from .models import UserProfile
    
    token = generate_verification_token()
    profile = user.profile
    profile.verification_token = token
    profile.save()
    
    verification_url = f"{request.scheme}://{request.get_host()}/core/verify-email/{token}/"
    
    subject = 'Verify Your Email Address'
    html_message = render_to_string('core/emails/verify_email.html', {
        'user': user,
        'verification_url': verification_url
    })
    plain_message = strip_tags(html_message)
    
    try:
        send_mail(
            subject,
            plain_message,
            settings.DEFAULT_FROM_EMAIL,
            [user.email],
            html_message=html_message,
            fail_silently=False
        )
        return True
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        return False

def send_contact_email(contact_data):
    """Send contact form email"""
    subject = f"Contact Form: {contact_data['subject']}"
    html_message = render_to_string('core/emails/contact_email.html', contact_data)
    plain_message = strip_tags(html_message)
    
    try:
        send_mail(
            subject,
            plain_message,
            settings.DEFAULT_FROM_EMAIL,
            [settings.CONTACT_EMAIL],
            html_message=html_message,
            fail_silently=False
        )
        return True
    except Exception as e:
        logger.error(f"Failed to send contact email: {e}")
        return False

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def validate_image_file(file):
    """Validate uploaded image file"""
    # Check file size
    if file.size > settings.MAX_IMAGE_SIZE * 1024 * 1024:
        return False, f"File size exceeds {settings.MAX_IMAGE_SIZE}MB"
    
    # Check file extension
    ext = os.path.splitext(file.name)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        return False, "Unsupported file format"
    
    # Try to open with PIL to verify it's an image
    try:
        img = Image.open(file)
        img.verify()
    except:
        return False, "Invalid image file"
    
    return True, "Valid image file"

def generate_unique_filename(instance, filename):
    """Generate unique filename for upload"""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4().hex}.{ext}"
    return filename

def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def parse_duration(duration_str):
    """Parse duration string to timedelta"""
    import re
    
    regex = re.compile(
        r'^((?P<days>[\.\d]+?)d)? *'
        r'((?P<hours>[\.\d]+?)h)? *'
        r'((?P<minutes>[\.\d]+?)m)? *'
        r'((?P<seconds>[\.\d]+?)s)?$'
    )
    
    parts = regex.match(duration_str)
    if not parts:
        return None
    
    parts = parts.groupdict()
    time_params = {}
    for name, param in parts.items():
        if param:
            time_params[name] = float(param)
    
    return timedelta(**time_params)

def calculate_storage_usage(user):
    """Calculate storage usage for a user"""
    from dataset_manager.models import Dataset, Image
    from prediction.models import Prediction
    
    total_size = 0
    
    # Calculate dataset sizes
    datasets = Dataset.objects.filter(uploaded_by=user)
    for dataset in datasets:
        if dataset.zip_file:
            try:
                total_size += dataset.zip_file.size
            except:
                pass
    
    # Calculate image sizes
    images = Image.objects.filter(dataset__uploaded_by=user)
    for image in images:
        if image.file:
            try:
                total_size += image.file.size
            except:
                pass
    
    # Calculate prediction sizes
    predictions = Prediction.objects.filter(created_by=user)
    for prediction in predictions:
        if prediction.image:
            try:
                total_size += prediction.image.size
            except:
                pass
    
    return total_size

def check_storage_limit(user):
    """Check if user has exceeded storage limit"""
    from .models import SiteSettings
    
    settings = SiteSettings.get_settings()
    used_storage = calculate_storage_usage(user)
    limit_bytes = settings.max_user_storage * 1024 * 1024 * 1024  # Convert GB to bytes
    
    return used_storage < limit_bytes, used_storage, limit_bytes

def create_activity_log(user, action, description, request=None, **kwargs):
    """Create activity log entry"""
    from .models import ActivityLog
    
    return ActivityLog.log_activity(user, action, description, request, **kwargs)

def create_notification(user, title, message, notification_type='info', link='', expires_in_days=None):
    """Create user notification"""
    from .models import Notification
    
    return Notification.create_notification(
        user, title, message, notification_type, link, expires_in_days
    )

def get_model_choices():
    """Get available ML model choices"""
    return [
        ('cnn', 'Custom CNN'),
        ('vgg16', 'VGG16'),
        ('resnet50', 'ResNet50'),
        ('inceptionv3', 'InceptionV3'),
    ]

def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate accuracy metrics for model evaluation"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }

def generate_chart_colors(num_colors):
    """Generate chart colors for visualization"""
    colors = [
        '#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8',
        '#6c757d', '#6610f2', '#e83e8c', '#fd7e14', '#20c997'
    ]
    
    if num_colors <= len(colors):
        return colors[:num_colors]
    
    # Generate additional colors if needed
    import colorsys
    additional = []
    for i in range(num_colors - len(colors)):
        hue = i / (num_colors - len(colors))
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        additional.append(color)
    
    return colors + additional