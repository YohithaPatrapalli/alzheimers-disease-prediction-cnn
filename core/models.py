from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os

class UserProfile(models.Model):
    """Extended user profile model"""
    
    USER_TYPES = [
        ('admin', 'Administrator'),
        ('doctor', 'Doctor'),
        ('researcher', 'Researcher'),
        ('technician', 'Technician'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    user_type = models.CharField(max_length=20, choices=USER_TYPES, default='doctor')
    institution = models.CharField(max_length=200, blank=True)
    license_number = models.CharField(max_length=50, blank=True)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    phone = models.CharField(max_length=20, blank=True)
    department = models.CharField(max_length=100, blank=True)
    specialization = models.CharField(max_length=200, blank=True)
    bio = models.TextField(blank=True, max_length=500)
    date_of_birth = models.DateField(null=True, blank=True)
    address = models.TextField(blank=True, max_length=500)
    city = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, blank=True)
    postal_code = models.CharField(max_length=20, blank=True)
    
    # Email notifications preferences
    email_notifications = models.BooleanField(default=True)
    prediction_alerts = models.BooleanField(default=True)
    model_training_alerts = models.BooleanField(default=True)
    newsletter_subscription = models.BooleanField(default=False)
    
    # Two-factor authentication
    two_factor_enabled = models.BooleanField(default=False)
    two_factor_secret = models.CharField(max_length=32, blank=True, null=True)
    
    # Account status
    is_verified = models.BooleanField(default=False)
    verification_token = models.CharField(max_length=100, blank=True, null=True)
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    last_activity = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
        indexes = [
            models.Index(fields=['user_type']),
            models.Index(fields=['is_verified']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.get_user_type_display()}"
    
    def get_full_name(self):
        return self.user.get_full_name() or self.user.username
    
    def get_avatar_url(self):
        if self.avatar and hasattr(self.avatar, 'url'):
            return self.avatar.url
        return None
    
    def update_last_activity(self):
        self.last_activity = timezone.now()
        self.save(update_fields=['last_activity'])

class ActivityLog(models.Model):
    """User activity logging model"""
    
    ACTION_TYPES = [
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('upload', 'Upload'),
        ('prediction', 'Prediction'),
        ('training', 'Training'),
        ('export', 'Export'),
        ('delete', 'Delete'),
        ('update', 'Update'),
        ('create', 'Create'),
        ('view', 'View'),
        ('download', 'Download'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    action = models.CharField(max_length=50, choices=ACTION_TYPES)
    description = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    request_path = models.CharField(max_length=500, blank=True)
    request_method = models.CharField(max_length=10, blank=True)
    status_code = models.IntegerField(null=True, blank=True)
    response_time = models.FloatField(null=True, blank=True, help_text="Response time in seconds")
    additional_data = models.JSONField(default=dict, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Activity Log'
        verbose_name_plural = 'Activity Logs'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['action']),
            models.Index(fields=['timestamp']),
        ]
    
    def __str__(self):
        return f"{self.user.username if self.user else 'Anonymous'} - {self.action} - {self.timestamp}"
    
    @classmethod
    def log_activity(cls, user, action, description, request=None, **kwargs):
        """Helper method to log activity"""
        log_entry = cls(
            user=user,
            action=action,
            description=description,
            additional_data=kwargs
        )
        
        if request:
            log_entry.ip_address = request.META.get('REMOTE_ADDR')
            log_entry.user_agent = request.META.get('HTTP_USER_AGENT', '')
            log_entry.request_path = request.path
            log_entry.request_method = request.method
        
        log_entry.save()
        return log_entry

class SiteSettings(models.Model):
    """Global site settings model"""
    
    site_name = models.CharField(max_length=100, default='Alzheimer\'s Disease Prediction System')
    site_description = models.TextField(blank=True, max_length=500)
    logo = models.ImageField(upload_to='settings/', null=True, blank=True)
    favicon = models.ImageField(upload_to='settings/', null=True, blank=True)
    
    # Appearance
    primary_color = models.CharField(max_length=7, default='#007bff')
    secondary_color = models.CharField(max_length=7, default='#6c757d')
    accent_color = models.CharField(max_length=7, default='#28a745')
    dark_mode_enabled = models.BooleanField(default=False)
    custom_css = models.TextField(blank=True)
    
    # Features
    enable_registration = models.BooleanField(default=True)
    enable_analytics = models.BooleanField(default=True)
    maintenance_mode = models.BooleanField(default=False)
    maintenance_message = models.TextField(blank=True, default='Site is under maintenance. Please check back later.')
    
    # Security
    max_login_attempts = models.IntegerField(default=5)
    session_timeout = models.IntegerField(default=30, help_text="Session timeout in minutes")
    password_expiry_days = models.IntegerField(default=90, help_text="Password expiry in days")
    two_factor_required = models.BooleanField(default=False)
    
    # Email settings
    contact_email = models.EmailField(default='admin@example.com')
    support_email = models.EmailField(default='support@example.com')
    noreply_email = models.EmailField(default='noreply@example.com')
    
    # API settings
    api_rate_limit = models.IntegerField(default=100, help_text="API requests per minute")
    api_enabled = models.BooleanField(default=True)
    
    # ML Model settings
    default_model = models.CharField(max_length=50, default='cnn')
    confidence_threshold = models.FloatField(default=0.85, help_text="Minimum confidence for predictions")
    batch_size = models.IntegerField(default=32)
    max_image_size = models.IntegerField(default=10, help_text="Maximum image size in MB")
    
    # Storage limits (in GB)
    max_dataset_size = models.IntegerField(default=10, help_text="Maximum dataset size in GB")
    max_user_storage = models.IntegerField(default=5, help_text="Maximum storage per user in GB")
    
    # Social media links
    facebook_url = models.URLField(blank=True)
    twitter_url = models.URLField(blank=True)
    linkedin_url = models.URLField(blank=True)
    github_url = models.URLField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Site Settings'
        verbose_name_plural = 'Site Settings'
    
    def __str__(self):
        return self.site_name
    
    def save(self, *args, **kwargs):
        # Ensure only one instance exists
        if not self.pk and SiteSettings.objects.exists():
            return
        super().save(*args, **kwargs)
    
    @classmethod
    def get_settings(cls):
        """Get or create settings instance"""
        settings, created = cls.objects.get_or_create(pk=1)
        return settings

class Notification(models.Model):
    """User notifications model"""
    
    NOTIFICATION_TYPES = [
        ('info', 'Information'),
        ('success', 'Success'),
        ('warning', 'Warning'),
        ('error', 'Error'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    title = models.CharField(max_length=200)
    message = models.TextField()
    notification_type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES, default='info')
    link = models.CharField(max_length=500, blank=True)
    is_read = models.BooleanField(default=False)
    read_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['is_read']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.title}"
    
    def mark_as_read(self):
        if not self.is_read:
            self.is_read = True
            self.read_at = timezone.now()
            self.save(update_fields=['is_read', 'read_at'])
    
    @classmethod
    def mark_all_read(cls, user):
        cls.objects.filter(user=user, is_read=False).update(
            is_read=True, 
            read_at=timezone.now()
        )
    
    @classmethod
    def create_notification(cls, user, title, message, notification_type='info', link='', expires_in_days=None):
        notification = cls(
            user=user,
            title=title,
            message=message,
            notification_type=notification_type,
            link=link
        )
        
        if expires_in_days:
            notification.expires_at = timezone.now() + timezone.timedelta(days=expires_in_days)
        
        notification.save()
        return notification

class SystemMetric(models.Model):
    """System performance metrics"""
    
    METRIC_TYPES = [
        ('cpu', 'CPU Usage'),
        ('memory', 'Memory Usage'),
        ('disk', 'Disk Usage'),
        ('api', 'API Calls'),
        ('predictions', 'Predictions'),
        ('training', 'Training Time'),
    ]
    
    metric_type = models.CharField(max_length=20, choices=METRIC_TYPES)
    value = models.FloatField()
    unit = models.CharField(max_length=20, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['metric_type', '-timestamp']),
        ]
    
    def __str__(self):
        return f"{self.metric_type}: {self.value} {self.unit} @ {self.timestamp}"
    
    @classmethod
    def record_metric(cls, metric_type, value, unit=''):
        return cls.objects.create(
            metric_type=metric_type,
            value=value,
            unit=unit
        )