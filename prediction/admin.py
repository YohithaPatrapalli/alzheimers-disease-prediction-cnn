import django.contrib  # pyright: ignore[reportMissingModuleSource]
import django.utils.html  # pyright: ignore[reportMissingModuleSource]
from django.urls import reverse  # pyright: ignore[reportMissingModuleSource]
from .models import TrainedModel, Prediction, BatchPrediction, ModelVersion, PredictionFeedback


@django.contrib.admin.register(TrainedModel)
class TrainedModelAdmin(django.contrib.admin.ModelAdmin):
    list_display = ['name', 'model_type', 'get_status_badge', 'dataset', 'accuracy', 'created_by', 'created_at']
    list_filter = ['model_type', 'status', 'created_at']
    search_fields = ['name', 'description', 'created_by__username']
    readonly_fields = ['created_at', 'updated_at', 'get_model_summary']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'model_type', 'dataset')
        }),
        ('Model File', {
            'fields': ('file', 'get_model_summary')
        }),
        ('Performance Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score', 'auc_score', 'confusion_matrix')
        }),
        ('Training Info', {
            'fields': ('status', 'training_time', 'epochs_trained', 'created_by')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def get_status_badge(self, obj):
        colors = {
            'training': 'warning',
            'completed': 'success',
            'failed': 'danger',
            'pending': 'secondary'
        }
        return django.utils.html.format_html(
            '<span class="badge bg-{}">{}</span>',
            colors.get(obj.status, 'secondary'),
            obj.get_status_display()
        )
    get_status_badge.short_description = 'Status'
    
    def get_model_summary(self, obj):
        if obj.model_summary:
            return django.utils.html.format_html('<pre style="max-height: 200px; overflow: auto;">{}</pre>', obj.model_summary)
        return django.utils.html.format_html('<span class="text-muted">No summary available</span>')
    get_model_summary.short_description = 'Model Architecture'
    
    def save_model(self, request, obj, form, change):
        if not change:  # If creating new object
            obj.created_by = request.user
        super().save_model(request, obj, form, change)


@django.contrib.admin.register(Prediction)
class PredictionAdmin(django.contrib.admin.ModelAdmin):
    list_display = ['id', 'get_image_preview', 'predicted_stage', 'confidence_display', 'model', 'get_user', 'created_at']
    list_filter = ['predicted_stage', 'created_at']
    search_fields = ['created_by__username', 'session_id']
    readonly_fields = ['get_image_preview', 'get_processed_preview', 'get_qr_code_preview', 'created_at']
    
    fieldsets = (
        ('Input Image', {
            'fields': ('image', 'get_image_preview')
        }),
        ('Processed Image', {
            'fields': ('processed_image', 'get_processed_preview')
        }),
        ('Prediction Results', {
            'fields': ('model', 'predicted_stage', 'confidence', 'confidence_scores')
        }),
        ('QR Code', {
            'fields': ('qr_code', 'get_qr_code_preview')
        }),
        ('Metadata', {
            'fields': ('processing_time', 'created_by', 'session_id', 'created_at')
        })
    )
    
    def get_image_preview(self, obj):
        if obj.image:
            return django.utils.html.format_html(
                '<img src="{}" style="max-height: 100px; max-width: 100px; border-radius: 5px;" />',
                obj.image.url
            )
        return django.utils.html.format_html('<span class="text-muted">No image</span>')
    get_image_preview.short_description = 'Original Image'
    
    def get_processed_preview(self, obj):
        if obj.processed_image:
            return django.utils.html.format_html(
                '<img src="{}" style="max-height: 100px; max-width: 100px; border-radius: 5px;" />',
                obj.processed_image.url
            )
        return django.utils.html.format_html('<span class="text-muted">No processed image</span>')
    get_processed_preview.short_description = 'Processed Image'
    
    def get_qr_code_preview(self, obj):
        if obj.qr_code:
            return django.utils.html.format_html(
                '<img src="{}" style="max-height: 100px; max-width: 100px;" />',
                obj.qr_code.url
            )
        return django.utils.html.format_html('<span class="text-muted">No QR code</span>')
    get_qr_code_preview.short_description = 'QR Code'
    
    def confidence_display(self, obj):
        color = 'success' if obj.confidence >= 85 else 'warning' if obj.confidence >= 70 else 'danger'
        return django.utils.html.format_html(
            '<span class="badge bg-{}">{:.1f}%</span>',
            color, obj.confidence
        )
    confidence_display.short_description = 'Confidence'
    
    def get_user(self, obj):
        if obj.created_by:
            return obj.created_by.username
        return 'Anonymous'
    get_user.short_description = 'User'


@django.contrib.admin.register(BatchPrediction)
class BatchPredictionAdmin(django.contrib.admin.ModelAdmin):
    list_display = ['name', 'model', 'get_status_badge', 'total_images', 'processed_images', 'get_user', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['name', 'created_by__username']
    readonly_fields = ['created_at', 'completed_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'model', 'zip_file')
        }),
        ('Status', {
            'fields': ('status', 'total_images', 'processed_images', 'results_file')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'completed_at', 'created_by')
        })
    )
    
    def get_status_badge(self, obj):
        colors = {
            'pending': 'secondary',
            'processing': 'info',
            'completed': 'success',
            'failed': 'danger'
        }
        return django.utils.html.format_html(
            '<span class="badge bg-{}">{}</span>',
            colors.get(obj.status, 'secondary'),
            obj.get_status_display()
        )
    get_status_badge.short_description = 'Status'
    
    def get_user(self, obj):
        if obj.created_by:
            return obj.created_by.username
        return 'Anonymous'
    get_user.short_description = 'Created By'


@django.contrib.admin.register(ModelVersion)
class ModelVersionAdmin(django.contrib.admin.ModelAdmin):
    list_display = ['model', 'version', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['model__name', 'version']
    readonly_fields = ['created_at']


@django.contrib.admin.register(PredictionFeedback)
class PredictionFeedbackAdmin(django.contrib.admin.ModelAdmin):
    list_display = ['prediction', 'rating', 'correct_stage', 'get_user', 'created_at']
    list_filter = ['rating', 'correct_stage', 'created_at']
    search_fields = ['prediction__id', 'comments', 'user__username']
    readonly_fields = ['created_at']
    
    fieldsets = (
        ('Feedback Information', {
            'fields': ('prediction', 'user', 'rating', 'correct_stage', 'comments')
        }),
        ('Model Improvement', {
            'fields': ('is_helpful', 'used_for_retraining'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )
    
    def get_user(self, obj):
        if obj.user:
            return obj.user.username
        return 'Anonymous'
    get_user.short_description = 'User'