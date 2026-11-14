"""
Signals for prediction app
"""

from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.core.files.storage import default_storage
import os

from .models import Prediction, TrainedModel, BatchPrediction


@receiver(post_save, sender=Prediction)
def prediction_post_save(sender, instance, created, **kwargs):
    """
    Handle actions after prediction is saved
    """
    if created:
        # Update model performance metrics
        if instance.model:
            update_model_performance(instance.model)


@receiver(post_delete, sender=Prediction)
def prediction_post_delete(sender, instance, **kwargs):
    """
    Clean up files when prediction is deleted
    """
    # Delete image files
    if instance.image and default_storage.exists(instance.image.path):
        try:
            default_storage.delete(instance.image.path)
        except:
            pass
    
    if instance.processed_image and default_storage.exists(instance.processed_image.path):
        try:
            default_storage.delete(instance.processed_image.path)
        except:
            pass
    
    if instance.qr_code and default_storage.exists(instance.qr_code.path):
        try:
            default_storage.delete(instance.qr_code.path)
        except:
            pass


@receiver(post_save, sender=TrainedModel)
def model_post_save(sender, instance, created, **kwargs):
    """
    Handle actions after model is saved
    """
    if created:
        # Model was created
        pass


@receiver(post_delete, sender=TrainedModel)
def model_post_delete(sender, instance, **kwargs):
    """
    Clean up model file when model is deleted
    """
    if instance.file and default_storage.exists(instance.file.path):
        try:
            default_storage.delete(instance.file.path)
        except:
            pass


@receiver(post_save, sender=BatchPrediction)
def batch_prediction_post_save(sender, instance, created, **kwargs):
    """
    Handle actions after batch prediction is saved
    """
    if created:
        # Start batch processing
        from .tasks import batch_prediction_task
        batch_prediction_task.delay(instance.id)


def update_model_performance(model):
    """
    Update model performance metrics based on recent predictions
    """
    from django.utils import timezone
    from datetime import timedelta
    
    # Get predictions from last 30 days
    cutoff = timezone.now() - timedelta(days=30)
    predictions = Prediction.objects.filter(
        model=model,
        created_at__gte=cutoff
    )
    
    if predictions.exists():
        # Update average confidence
        from django.db.models import Avg
        avg_confidence = predictions.aggregate(Avg('confidence'))['confidence__avg']
        
        # Update model performance log
        from .models import ModelPerformanceLog
        ModelPerformanceLog.objects.update_or_create(
            model=model,
            date=timezone.localdate(),
            defaults={
                'predictions_count': predictions.count(),
                'average_confidence': avg_confidence,
            },
        )