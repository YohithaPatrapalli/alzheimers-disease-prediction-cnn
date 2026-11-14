from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
import json
import os

# Import from dataset_manager - these models exist
from dataset_manager.models import Dataset, DatasetSplit, ImageCategory


class TrainedModel(models.Model):
    """
    Trained ML model for Alzheimer's disease classification
    """
    
    MODEL_TYPES = [
        ('cnn', 'Custom CNN'),
        ('vgg16', 'VGG16'),
        ('vgg19', 'VGG19'),
        ('resnet50', 'ResNet50'),
        ('resnet101', 'ResNet101'),
        ('inceptionv3', 'InceptionV3'),
        ('inceptionresnetv2', 'InceptionResNetV2'),
        ('densenet121', 'DenseNet121'),
        ('densenet169', 'DenseNet169'),
        ('mobilenet', 'MobileNet'),
        ('mobilenetv2', 'MobileNetV2'),
        ('efficientnetb0', 'EfficientNetB0'),
        ('efficientnetb7', 'EfficientNetB7'),
        ('xception', 'Xception'),
        ('nasnetmobile', 'NASNetMobile')
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    name = models.CharField(max_length=200, db_index=True)
    description = models.TextField(blank=True)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES, db_index=True)
    
    # Model file
    file = models.FileField(upload_to='models/', null=True, blank=True)
    model_summary = models.TextField(blank=True, help_text="Model architecture summary")
    model_config = models.JSONField(default=dict, blank=True)
    
    # Training data
    dataset = models.ForeignKey(
        Dataset, 
        on_delete=models.SET_NULL, 
        null=True, 
        related_name='trained_models'
    )
    dataset_split = models.ForeignKey(
        DatasetSplit, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    
    # Training configuration
    training_config = models.JSONField(default=dict, blank=True)
    epochs_trained = models.IntegerField(default=0)
    training_time = models.FloatField(null=True, blank=True, help_text="Training time in seconds")
    
    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])
    precision = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])
    recall = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])
    f1_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])
    auc_score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])
    
    # Detailed metrics
    confusion_matrix = models.JSONField(null=True, blank=True)
    classification_report = models.JSONField(null=True, blank=True)
    per_class_metrics = models.JSONField(null=True, blank=True)
    training_history = models.JSONField(null=True, blank=True)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', db_index=True)
    error_message = models.TextField(blank=True, null=True)
    
    # Version control
    version = models.CharField(max_length=20, blank=True)
    parent_model = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='child_models')
    
    # Active flag for current model
    is_active = models.BooleanField(default=False, db_index=True, help_text="Whether this model is currently active for predictions")
    
    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='trained_models')
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['model_type', 'status']),
            models.Index(fields=['created_by', '-created_at']),
            models.Index(fields=['is_active']),  # Add index for is_active
        ]
    
    def __str__(self):
        return f"{self.name} ({self.get_model_type_display()})"
    
    def get_metrics_display(self):
        """Get formatted metrics for display"""
        return {
            'accuracy': f"{self.accuracy*100:.2f}%" if self.accuracy else 'N/A',
            'precision': f"{self.precision*100:.2f}%" if self.precision else 'N/A',
            'recall': f"{self.recall*100:.2f}%" if self.recall else 'N/A',
            'f1_score': f"{self.f1_score*100:.2f}%" if self.f1_score else 'N/A',
            'auc_score': f"{self.auc_score*100:.2f}%" if self.auc_score else 'N/A'
        }
    
    def get_model_size(self):
        """Get model file size"""
        if self.file and os.path.exists(self.file.path):
            size_bytes = os.path.getsize(self.file.path)
            return self._format_size(size_bytes)
        return "N/A"
    
    def _format_size(self, size_bytes):
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def get_latest_version(self):
        """Get latest version number"""
        if self.version:
            return self.version
        return "1.0.0"
    
    def is_deployable(self):
        """Check if model is deployable"""
        return self.status == 'completed' and self.accuracy and self.accuracy >= 0.7


class ModelVersion(models.Model):
    """Model version tracking"""
    
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name='versions')
    version = models.CharField(max_length=20)
    file = models.FileField(upload_to='models/versions/')
    metrics = models.JSONField(default=dict)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['model', 'version']
    
    def __str__(self):
        return f"{self.model.name} v{self.version}"


class Prediction(models.Model):
    """
    Single image prediction result
    """
    
    STAGE_CHOICES = [
        ('non_demented', 'Non-Demented'),
        ('very_mild', 'Very Mild Demented'),
        ('mild', 'Mild Demented'),
        ('moderate', 'Moderate Demented'),
    ]
    
    # Image files
    image = models.ImageField(upload_to='predictions/input/')
    processed_image = models.ImageField(upload_to='predictions/processed/', null=True, blank=True)
    
    # Prediction details
    model = models.ForeignKey(TrainedModel, on_delete=models.SET_NULL, null=True, related_name='predictions')
    predicted_stage = models.CharField(max_length=20, choices=STAGE_CHOICES, db_index=True)
    confidence = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    confidence_scores = models.JSONField(default=dict, help_text="Confidence scores for all classes")
    
    # Explanation/Visualization
    explanation = models.JSONField(null=True, blank=True, help_text="Grad-CAM or other explanation data")
    
    # QR code
    qr_code = models.ImageField(upload_to='qrcodes/', null=True, blank=True)
    qr_data = models.TextField(blank=True, help_text="Data encoded in QR code")
    
    # Metadata
    processing_time = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")
    image_metadata = models.JSONField(default=dict, blank=True, help_text="Original image metadata")
    
    # User info
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='predictions')
    session_id = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['predicted_stage', '-created_at']),
            models.Index(fields=['created_by', '-created_at']),
            models.Index(fields=['session_id']),
        ]
    
    def __str__(self):
        return f"Prediction {self.id} - {self.get_predicted_stage_display()} ({self.confidence:.1f}%)"
    
    def get_confidence_color(self):
        """Get color based on confidence level"""
        if self.confidence >= 85:
            return 'success'
        elif self.confidence >= 70:
            return 'warning'
        else:
            return 'danger'
    
    def get_stage_class(self):
        """Get CSS class for stage"""
        classes = {
            'non_demented': 'success',
            'very_mild': 'info',
            'mild': 'warning',
            'moderate': 'danger'
        }
        return classes.get(self.predicted_stage, 'secondary')
    
    def get_top_predictions(self, n=3):
        """Get top n predictions by confidence"""
        sorted_scores = sorted(
            self.confidence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        return [{'stage': k, 'confidence': v} for k, v in sorted_scores]


class BatchPrediction(models.Model):
    """
    Batch prediction for multiple images
    """
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    name = models.CharField(max_length=200)
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name='batch_predictions')
    
    # Files
    zip_file = models.FileField(upload_to='batch_predictions/input/')
    results_file = models.FileField(upload_to='batch_predictions/results/', null=True, blank=True)
    results_summary = models.JSONField(default=dict, blank=True)
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', db_index=True)
    total_images = models.IntegerField(default=0)
    processed_images = models.IntegerField(default=0)
    failed_images = models.IntegerField(default=0)
    
    # Timing
    processing_time = models.FloatField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # User
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='batch_predictions')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.status}"
    
    def get_progress(self):
        """Get progress percentage"""
        if self.total_images == 0:
            return 0
        return int((self.processed_images / self.total_images) * 100)
    
    def get_summary_stats(self):
        """Get summary statistics"""
        if self.results_summary:
            return self.results_summary
        return {
            'total': self.total_images,
            'processed': self.processed_images,
            'failed': self.failed_images,
            'success_rate': (self.processed_images / self.total_images * 100) if self.total_images > 0 else 0
        }


class PredictionFeedback(models.Model):
    """
    User feedback on predictions
    """
    
    RATING_CHOICES = [(i, str(i)) for i in range(1, 6)]
    
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, related_name='feedback')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='prediction_feedback')
    
    rating = models.IntegerField(choices=RATING_CHOICES, validators=[MinValueValidator(1), MaxValueValidator(5)])
    correct_stage = models.CharField(max_length=20, choices=Prediction.STAGE_CHOICES, null=True, blank=True)
    comments = models.TextField(blank=True)
    
    # For model improvement
    is_helpful = models.BooleanField(default=True)
    used_for_retraining = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['prediction', 'user']
    
    def __str__(self):
        return f"Feedback for Prediction {self.prediction.id} - Rating: {self.rating}"


class ModelPerformanceLog(models.Model):
    """
    Log model performance over time
    """
    
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name='performance_logs')
    predictions_count = models.IntegerField(default=0)
    average_confidence = models.FloatField(null=True, blank=True)
    average_processing_time = models.FloatField(null=True, blank=True)
    stage_distribution = models.JSONField(default=dict)
    date = models.DateField(default=timezone.localdate)
    
    class Meta:
        ordering = ['-date']
        unique_together = ['model', 'date']
    
    def __str__(self):
        return f"{self.model.name} - {self.date}"