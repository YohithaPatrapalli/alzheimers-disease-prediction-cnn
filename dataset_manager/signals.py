"""
Signals for dataset_manager app
"""

from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.core.files.storage import default_storage
from .models import Dataset, Image


@receiver(post_delete, sender=Dataset)
def dataset_post_delete(sender, instance, **kwargs):
    """
    Clean up dataset files when dataset is deleted
    """
    # Delete zip file
    if instance.zip_file and default_storage.exists(instance.zip_file.path):
        try:
            default_storage.delete(instance.zip_file.path)
        except:
            pass
    
    # Delete all associated images
    for image in instance.images.all():
        if image.file and default_storage.exists(image.file.path):
            try:
                default_storage.delete(image.file.path)
            except:
                pass


@receiver(post_delete, sender=Image)
def image_post_delete(sender, instance, **kwargs):
    """
    Clean up image files when image is deleted
    """
    if instance.file and default_storage.exists(instance.file.path):
        try:
            default_storage.delete(instance.file.path)
        except:
            pass
    
    if instance.processed_file and default_storage.exists(instance.processed_file.path):
        try:
            default_storage.delete(instance.processed_file.path)
        except:
            pass