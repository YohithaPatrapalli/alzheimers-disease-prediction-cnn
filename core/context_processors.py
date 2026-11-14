from django.conf import settings
from .models import SiteSettings

def site_settings(request):
    """
    Context processor to add site settings to all templates.
    """
    try:
        settings_obj = SiteSettings.objects.first()
    except:
        settings_obj = None
    
    return {
        'site_settings': settings_obj,
        'DEBUG': settings.DEBUG,
        'APP_VERSION': getattr(settings, 'APP_VERSION', '1.0.0')
    }

def user_notifications(request):
    """
    Context processor to add user notifications to templates.
    """
    notifications = []
    unread_count = 0
    
    if request.user.is_authenticated:
        # Get user notifications (you can implement this based on your notification system)
        # This is a placeholder for demonstration
        notifications = [
            {'message': 'Welcome back!', 'type': 'info'},
            {'message': 'Your model training is complete', 'type': 'success'}
        ]
        unread_count = len(notifications)
    
    return {
        'notifications': notifications,
        'unread_notifications': unread_count
    }

def breadcrumbs(request):
    """
    Context processor to generate breadcrumbs based on request path.
    """
    path = request.path.split('/')[1:]
    breadcrumbs = []
    url = ''
    
    for part in path:
        if part and part not in ['static', 'media']:
            url += f'/{part}'
            breadcrumbs.append({
                'name': part.replace('-', ' ').title(),
                'url': url
            })
    
    return {
        'breadcrumbs': breadcrumbs
    }