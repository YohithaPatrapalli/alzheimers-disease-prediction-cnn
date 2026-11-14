from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from .models import Dataset, ImageCategory, Image, DatasetSplit

class ImageCategoryInline(admin.TabularInline):
    model = ImageCategory
    extra = 1
    readonly_fields = ['count']

class ImageInline(admin.TabularInline):
    model = Image
    extra = 0
    readonly_fields = ['original_filename', 'file', 'processed', 'created_at']
    fields = ['original_filename', 'category', 'processed', 'created_at']
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'get_status_badge', 'uploaded_by', 'total_images', 'get_progress', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['name', 'description', 'uploaded_by__username']
    readonly_fields = ['total_images', 'created_at', 'updated_at', 'get_dataset_preview']
    inlines = [ImageCategoryInline, ImageInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'zip_file', 'uploaded_by')
        }),
        ('Status', {
            'fields': ('status', 'total_images', 'get_dataset_preview')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    def get_status_badge(self, obj):
        colors = {
            'uploading': 'warning',
            'processing': 'info',
            'ready': 'success',
            'failed': 'danger'
        }
        return format_html(
            '<span class="badge bg-{}">{}</span>',
            colors.get(obj.status, 'secondary'),
            obj.get_status_display()
        )
    get_status_badge.short_description = 'Status'
    
    def get_progress(self, obj):
        progress = obj.get_progress_percentage()
        return format_html(
            '<div class="progress" style="height: 20px;">'
            '<div class="progress-bar" role="progressbar" style="width: {}%;" '
            'aria-valuenow="{}" aria-valuemin="0" aria-valuemax="100">{}%</div>'
            '</div>',
            progress, progress, progress
        )
    get_progress.short_description = 'Progress'
    
    def get_dataset_preview(self, obj):
        if obj.pk:
            categories = obj.categories.all()
            if categories:
                html = '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
                for category in categories[:4]:
                    html += f'''
                    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; min-width: 150px;">
                        <strong>{category.get_category_display()}</strong><br>
                        <span class="badge bg-info">{category.count} images</span>
                    </div>
                    '''
                html += '</div>'
                return format_html(html)
        return format_html('<span class="text-muted">No data available</span>')
    get_dataset_preview.short_description = 'Dataset Preview'
    
    def save_model(self, request, obj, form, change):
        if not change:  # If creating new object
            obj.uploaded_by = request.user
        super().save_model(request, obj, form, change)

@admin.register(ImageCategory)
class ImageCategoryAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'category', 'count']
    list_filter = ['category', 'dataset']
    search_fields = ['dataset__name', 'category']
    readonly_fields = ['count']

@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'dataset', 'category', 'original_filename', 'processed', 'get_image_preview', 'created_at']
    list_filter = ['processed', 'dataset', 'category']
    search_fields = ['original_filename', 'dataset__name']
    readonly_fields = ['get_image_preview', 'created_at']
    
    fieldsets = (
        ('Image Information', {
            'fields': ('dataset', 'category', 'original_filename')
        }),
        ('Files', {
            'fields': ('file', 'processed_file', 'get_image_preview')
        }),
        ('Status', {
            'fields': ('processed', 'created_at')
        })
    )
    
    def get_image_preview(self, obj):
        if obj.file:
            return format_html(
                '<img src="{}" style="max-height: 100px; max-width: 100px; border-radius: 5px;" />',
                obj.file.url
            )
        return format_html('<span class="text-muted">No image</span>')
    get_image_preview.short_description = 'Preview'

@admin.register(DatasetSplit)
class DatasetSplitAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'get_ratios', 'created_at']
    list_filter = ['created_at']
    search_fields = ['dataset__name']
    readonly_fields = ['created_at']
    
    def get_ratios(self, obj):
        return format_html(
            'Train: {}% | Val: {}% | Test: {}%',
            obj.train_ratio * 100,
            obj.val_ratio * 100,
            obj.test_ratio * 100
        )
    get_ratios.short_description = 'Split Ratios'