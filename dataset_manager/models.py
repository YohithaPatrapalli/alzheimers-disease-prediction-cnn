from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os
import uuid

class Dataset(models.Model):
    """
    Dataset model for storing uploaded MRI image datasets
    """
    
    STATUS_CHOICES = [
        ('uploading', 'Uploading'),
        ('processing', 'Processing'),
        ('ready', 'Ready'),
        ('failed', 'Failed'),
    ]
    
    name = models.CharField(max_length=200, db_index=True)
    description = models.TextField(blank=True)
    zip_file = models.FileField(upload_to='datasets/archives/')
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='uploaded_datasets')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploading', db_index=True)
    total_images = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, null=True)
    processing_time = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['name', 'status']),
            models.Index(fields=['uploaded_by', '-created_at']),
        ]
    
    def __str__(self):
        return self.name
    
    def get_progress_percentage(self):
        """Get processing progress percentage"""
        if self.total_images == 0:
            return 0
        processed = self.images.filter(processed=True).count()
        return int((processed / self.total_images) * 100)
    
    def get_category_distribution(self):
        """Get distribution of images across categories"""
        categories = self.categories.all()
        return {
            cat.category: cat.count 
            for cat in categories
        }
    
    def get_processed_count(self):
        """Get count of processed images"""
        return self.images.filter(processed=True).count()
    
    def get_total_size(self):
        """Calculate total size of dataset"""
        total = 0
        if self.zip_file:
            try:
                total += self.zip_file.size
            except:
                pass
        
        for image in self.images.all():
            if image.file:
                try:
                    total += image.file.size
                except:
                    pass
        
        return total
    
    def get_formatted_size(self):
        """Get formatted size string"""
        size_bytes = self.get_total_size()
        
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"

class ImageCategory(models.Model):
    """
    Category model for image classification
    """
    
    CATEGORY_CHOICES = [
        ('non_demented', 'Non-Demented'),
        ('very_mild', 'Very Mild Demented'),
        ('mild', 'Mild Demented'),
        ('moderate', 'Moderate Demented'),
    ]
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='categories')
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, db_index=True)
    count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['dataset', 'category']
        verbose_name = 'Image Category'
        verbose_name_plural = 'Image Categories'
        indexes = [
            models.Index(fields=['dataset', 'category']),
        ]
    
    def __str__(self):
        return f"{self.dataset.name} - {self.get_category_display()}"
    
    def increment_count(self):
        """Increment category count"""
        self.count += 1
        self.save(update_fields=['count'])
    
    def decrement_count(self):
        """Decrement category count"""
        if self.count > 0:
            self.count -= 1
            self.save(update_fields=['count'])

class Image(models.Model):
    """
    Individual MRI image model
    """
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='images')
    category = models.ForeignKey(ImageCategory, on_delete=models.SET_NULL, null=True, blank=True, related_name='images')
    original_filename = models.CharField(max_length=255)
    file = models.ImageField(upload_to='datasets/images/')
    processed = models.BooleanField(default=False, db_index=True)
    processed_file = models.ImageField(upload_to='datasets/processed/', null=True, blank=True)
    
    # Image metadata
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    channels = models.IntegerField(null=True, blank=True)
    file_size = models.IntegerField(null=True, blank=True, help_text="File size in bytes")
    format = models.CharField(max_length=10, null=True, blank=True)
    
    # Processing metadata
    processing_time = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")
    error_message = models.TextField(blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['dataset', 'category']),
            models.Index(fields=['processed']),
            models.Index(fields=['dataset', 'processed']),
        ]
    
    def __str__(self):
        return f"{self.original_filename} ({self.dataset.name})"
    
    def get_file_extension(self):
        """Get file extension"""
        return os.path.splitext(self.original_filename)[1].lower()
    
    def is_valid_image(self):
        """Check if image is valid"""
        return self.error_message is None
    
    def get_display_name(self):
        """Get display name (truncated if too long)"""
        if len(self.original_filename) > 30:
            return f"{self.original_filename[:27]}..."
        return self.original_filename
    
    def get_file_url(self):
        """Get file URL"""
        if self.processed and self.processed_file:
            return self.processed_file.url
        return self.file.url if self.file else None

class DatasetSplit(models.Model):
    """
    Dataset split configuration
    """
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='splits')
    train_ratio = models.FloatField(default=0.7, help_text="Training set ratio (0-1)")
    val_ratio = models.FloatField(default=0.15, help_text="Validation set ratio (0-1)")
    test_ratio = models.FloatField(default=0.15, help_text="Test set ratio (0-1)")
    
    train_path = models.CharField(max_length=500, blank=True)
    val_path = models.CharField(max_length=500, blank=True)
    test_path = models.CharField(max_length=500, blank=True)
    
    train_count = models.IntegerField(default=0)
    val_count = models.IntegerField(default=0)
    test_count = models.IntegerField(default=0)
    
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Dataset Split'
        verbose_name_plural = 'Dataset Splits'
    
    def __str__(self):
        return f"{self.dataset.name} Split ({self.created_at.date()})"
    
    def get_distribution(self):
        """Get split distribution as dictionary"""
        return {
            'train': self.train_count,
            'validation': self.val_count,
            'test': self.test_count
        }
    
    def get_percentages(self):
        """Get split percentages"""
        return {
            'train': self.train_ratio * 100,
            'validation': self.val_ratio * 100,
            'test': self.test_ratio * 100
        }
    
    def validate_split(self):
        """Validate split configuration"""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        return abs(total - 1.0) < 0.01

class DatasetProcessingLog(models.Model):
    """
    Log for dataset processing operations
    """
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='processing_logs')
    operation = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=[
        ('started', 'Started'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ])
    message = models.TextField(blank=True)
    progress = models.IntegerField(default=0, help_text="Progress percentage")
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"{self.dataset.name} - {self.operation} - {self.status}"
    
    def complete(self, success=True):
        """Mark log as completed"""
        self.status = 'completed' if success else 'failed'
        self.progress = 100 if success else self.progress
        self.completed_at = timezone.now()
        self.save()

class DatasetStatistics(models.Model):
    """
    Statistics for dataset analysis
    """
    
    dataset = models.OneToOneField(Dataset, on_delete=models.CASCADE, related_name='statistics')
    
    # Image statistics
    total_images = models.IntegerField(default=0)
    avg_width = models.FloatField(null=True, blank=True)
    avg_height = models.FloatField(null=True, blank=True)
    avg_file_size = models.FloatField(null=True, blank=True, help_text="Average file size in KB")
    
    # Category distribution
    category_distribution = models.JSONField(default=dict)
    
    # Processing statistics
    processing_time = models.FloatField(null=True, blank=True, help_text="Total processing time in seconds")
    success_rate = models.FloatField(null=True, blank=True, help_text="Percentage of successfully processed images")
    
    # Quality metrics
    quality_score = models.FloatField(null=True, blank=True, help_text="Overall dataset quality score (0-100)")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Dataset Statistics'
        verbose_name_plural = 'Dataset Statistics'
    
    def __str__(self):
        return f"Statistics for {self.dataset.name}"
    
    def update_statistics(self):
        """Update all statistics"""
        images = self.dataset.images.all()
        
        if images.exists():
            self.total_images = images.count()
            
            # Calculate averages
            widths = [img.width for img in images if img.width]
            heights = [img.height for img in images if img.height]
            sizes = [img.file_size for img in images if img.file_size]
            
            self.avg_width = sum(widths) / len(widths) if widths else None
            self.avg_height = sum(heights) / len(heights) if heights else None
            self.avg_file_size = sum(sizes) / len(sizes) if sizes else None
            
            # Category distribution
            distribution = {}
            for category in self.dataset.categories.all():
                distribution[category.category] = category.count
            self.category_distribution = distribution
            
            # Success rate
            successful = images.filter(error_message__isnull=True).count()
            self.success_rate = (successful / self.total_images) * 100 if self.total_images > 0 else 0
        
        self.save()