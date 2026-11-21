import qrcode
import json
import numpy as np
from io import BytesIO
from PIL import Image
import base64
from django.core.files.base import ContentFile
from django.conf import settings
from django.urls import reverse
import hashlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_PREDICTION_STAGE_ALIASES = {
    'nondemented': 'non_demented',
    'non_demented': 'non_demented',
    'verymild': 'very_mild',
    'very_mild': 'very_mild',
    'verymilddemented': 'very_mild',
    'mild': 'mild',
    'milddemented': 'mild',
    'milddementia': 'mild',
    'moderate': 'moderate',
    'moderatedemented': 'moderate',
    'moderatedementia': 'moderate',
    'moderateddementia': 'moderate',
}


def normalize_prediction_stage(stage):
    """Convert model class labels to the stage values used by Prediction."""
    normalized = ''.join(character for character in str(stage).casefold() if character.isalnum())
    try:
        return _PREDICTION_STAGE_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported prediction stage: {stage!r}") from error


def normalize_confidence_scores(scores):
    """Convert model class labels in confidence scores to Prediction stage values."""
    return {
        normalize_prediction_stage(stage): float(confidence)
        for stage, confidence in scores.items()
    }


def load_predictor_for_model(model):
    """Load a model and apply its saved or dataset-derived class order."""
    from ml_models.predict import Predictor
    from .models import Prediction

    predictor = Predictor()
    predictor.load_model(model.file.path)

    configured_names = (model.model_config or {}).get('class_names')
    if configured_names:
        predictor.class_names = configured_names
        return predictor

    config_path = Path(model.file.path).with_name(
        f'{Path(model.file.path).stem}_config.json'
    )
    if config_path.exists() or not model.dataset_split or not model.dataset_split.train_path:
        return predictor

    training_directory = Path(model.dataset_split.train_path)
    if not training_directory.is_absolute():
        training_directory = Path(settings.BASE_DIR) / training_directory
    if not training_directory.is_dir():
        logger.warning(
            "Cannot infer class order for model %s: training directory %s is missing",
            model.pk,
            training_directory,
        )
        return predictor

    stage_labels = dict(Prediction.STAGE_CHOICES)
    class_folders = sorted(
        folder.name for folder in training_directory.iterdir() if folder.is_dir()
    )
    try:
        class_names = [
            stage_labels[normalize_prediction_stage(folder)]
            for folder in class_folders
        ]
    except (KeyError, ValueError):
        logger.warning(
            "Cannot infer class order for model %s from folders %s",
            model.pk,
            class_folders,
        )
        return predictor

    output_count = predictor.model.output_shape[-1]
    if len(class_names) == output_count:
        predictor.class_names = class_names
    else:
        logger.warning(
            "Model %s has %s outputs but its training directory contains %s classes",
            model.pk,
            output_count,
            len(class_names),
        )

    return predictor


def generate_qr_code(data, size=10, border=4):
    """
    Generate QR code from data
    
    Args:
        data: Data to encode
        size: QR code size
        border: Border size
        
    Returns:
        PIL Image of QR code
    """
    qr = qrcode.QRCode(
        version=1,
        box_size=size,
        border=border
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    return img


def generate_prediction_qr(prediction, share_url=None, include_all_data=False):
    """Generate a QR code containing only the public report URL."""
    if share_url is None:
        share_url = reverse('prediction:print_report', args=[prediction.pk])
    return generate_qr_code(share_url)


def decode_qr_code(qr_image):
    """
    Decode QR code from image using OpenCV
    
    Args:
        qr_image: QR code image file
        
    Returns:
        Decoded data
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Read the image
        img = Image.open(qr_image)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array with proper dtype
        img_array = np.array(img, dtype=np.uint8)
        
        # Convert to grayscale for QR detection
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Use OpenCV QR code detector
        qr_detector = cv2.QRCodeDetector()
        data, bbox, _ = qr_detector.detectAndDecode(img_gray)
        
        if data:
            # Try to parse as JSON
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return {'raw_data': data}
        
        return None
    except Exception as e:
        logger.error(f"QR code decoding failed: {e}")
        return None


def calculate_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= 85:
        return 'success'
    elif confidence >= 70:
        return 'warning'
    else:
        return 'danger'


def get_stage_color(stage):
    """Get color for disease stage"""
    colors = {
        'non_demented': '#28a745',  # Green
        'very_mild': '#17a2b8',     # Teal
        'mild': '#ffc107',           # Yellow
        'moderate': '#dc3545'        # Red
    }
    return colors.get(stage, '#6c757d')


def prepare_prediction_chart_data(prediction):
    """Prepare data for confidence chart"""
    labels = []
    values = []
    colors = []
    
    for stage, confidence in prediction.confidence_scores.items():
        labels.append(stage.replace('_', ' ').title())
        values.append(round(confidence, 2))
        colors.append(get_stage_color(stage))
    
    return {
        'labels': labels,
        'values': values,
        'colors': colors
    }


def calculate_model_rank(model):
    """Calculate model rank based on multiple metrics"""
    if not model.accuracy:
        return 0
    
    score = model.accuracy * 0.4
    score += (model.precision or 0) * 0.2
    score += (model.recall or 0) * 0.2
    score += (model.f1_score or 0) * 0.2
    
    return round(score * 100, 2)


def get_best_model():
    """Get best performing model"""
    from .models import TrainedModel
    
    models = TrainedModel.objects.filter(status='completed')
    best_model = None
    best_score = 0
    
    for model in models:
        score = calculate_model_rank(model)
        if score > best_score:
            best_score = score
            best_model = model
    
    return best_model


def calculate_prediction_statistics(user=None, session_id=None, days=30):
    """Calculate prediction statistics"""
    from .models import Prediction
    from django.db.models import Count, Avg
    
    queryset = Prediction.objects.all()
    if user:
        queryset = queryset.filter(created_by=user)
    if session_id:
        queryset = queryset.filter(session_id=session_id)
    
    if days:
        cutoff = datetime.now() - timedelta(days=days)
        queryset = queryset.filter(created_at__gte=cutoff)
    
    total = queryset.count()
    
    if total == 0:
        return {
            'total': 0,
            'by_stage': {},
            'avg_confidence': 0,
            'success_rate': 0
        }
    
    # Stage distribution
    stage_counts = queryset.values('predicted_stage').annotate(
        count=Count('id')
    )
    
    by_stage = {}
    for item in stage_counts:
        stage = item['predicted_stage']
        by_stage[stage] = {
            'count': item['count'],
            'percentage': (item['count'] / total) * 100
        }
    
    # Average confidence
    avg_confidence = queryset.aggregate(Avg('confidence'))['confidence__avg']
    
    # Success rate (confidence > 70%)
    successful = queryset.filter(confidence__gte=70).count()
    success_rate = (successful / total) * 100
    
    return {
        'total': total,
        'by_stage': by_stage,
        'avg_confidence': round(avg_confidence, 2) if avg_confidence else 0,
        'success_rate': round(success_rate, 2)
    }


def export_predictions_to_csv(predictions):
    """Export predictions to CSV format"""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'ID', 'Date', 'Stage', 'Confidence (%)',
        'Model', 'Processing Time (s)', 'User'
    ])
    
    # Data
    for p in predictions:
        writer.writerow([
            p.id,
            p.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            p.get_predicted_stage_display(),
            round(p.confidence, 2),
            p.model.name if p.model else 'N/A',
            round(p.processing_time, 3) if p.processing_time else 'N/A',
            p.created_by.username if p.created_by else 'Anonymous'
        ])
    
    return output.getvalue()


def create_prediction_thumbnail(image_path, size=(100, 100)):
    """Create thumbnail for prediction image"""
    try:
        img = Image.open(image_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        thumb_io = BytesIO()
        img.save(thumb_io, format='JPEG', quality=85)
        thumb_io.seek(0)
        
        return ContentFile(thumb_io.getvalue(), 'thumbnail.jpg')
    except Exception as e:
        logger.error(f"Thumbnail creation failed: {e}")
        return None


def generate_prediction_report(prediction):
    """Generate HTML report for prediction"""
    from django.template.loader import render_to_string
    
    context = {
        'prediction': prediction,
        'chart_data': prepare_prediction_chart_data(prediction),
        'site_url': settings.SITE_URL
    }
    
    html = render_to_string('prediction/report_template.html', context)
    return html


def calculate_model_compatibility(model, image_size):
    """Check if model is compatible with image size"""
    if not model.model_config:
        return True
    
    expected_size = model.model_config.get('input_shape', [224, 224])[:2]
    
    if len(expected_size) == 2:
        return (abs(image_size[0] - expected_size[0]) < 10 and 
                abs(image_size[1] - expected_size[1]) < 10)
    
    return True


def get_model_suggestions(dataset_size, task_type='classification'):
    """Get model suggestions based on dataset size"""
    suggestions = []
    
    if dataset_size < 1000:
        suggestions = [
            {'name': 'MobileNetV2', 'type': 'mobilenetv2', 'speed': 'fast', 'accuracy': 'good'},
            {'name': 'Custom CNN', 'type': 'cnn', 'speed': 'fast', 'accuracy': 'good'}
        ]
    elif dataset_size < 5000:
        suggestions = [
            {'name': 'ResNet50', 'type': 'resnet50', 'speed': 'medium', 'accuracy': 'very good'},
            {'name': 'VGG16', 'type': 'vgg16', 'speed': 'slow', 'accuracy': 'good'},
            {'name': 'InceptionV3', 'type': 'inceptionv3', 'speed': 'medium', 'accuracy': 'very good'}
        ]
    else:
        suggestions = [
            {'name': 'EfficientNetB0', 'type': 'efficientnetb0', 'speed': 'fast', 'accuracy': 'excellent'},
            {'name': 'DenseNet121', 'type': 'densenet121', 'speed': 'medium', 'accuracy': 'excellent'},
            {'name': 'InceptionResNetV2', 'type': 'inceptionresnetv2', 'speed': 'slow', 'accuracy': 'excellent'}
        ]
    
    return suggestions