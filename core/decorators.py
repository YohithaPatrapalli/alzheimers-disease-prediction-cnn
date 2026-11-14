from django.shortcuts import redirect
from django.contrib import messages
from functools import wraps
from django.core.exceptions import PermissionDenied

def user_type_required(allowed_types):
    """
    Decorator to restrict access based on user type.
    Usage: @user_type_required(['doctor', 'admin'])
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                messages.error(request, 'Please login to access this page.')
                return redirect('login')
            
            user_type = getattr(request.user.profile, 'user_type', None)
            if user_type not in allowed_types:
                raise PermissionDenied("You don't have permission to access this page.")
            
            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator

def ajax_required(view_func):
    """
    Decorator to ensure the request is AJAX.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.headers.get('x-requested-with') == 'XMLHttpRequest':
            messages.error(request, 'This endpoint only accepts AJAX requests.')
            return redirect('core:dashboard')
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def log_activity(action=None):
    """
    Decorator to log user activity.
    Usage: @log_activity('page_view')
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            response = view_func(request, *args, **kwargs)
            
            if request.user.is_authenticated:
                from .models import ActivityLog
                ActivityLog.objects.create(
                    user=request.user,
                    action=action or request.resolver_match.url_name,
                    description=f"Accessed {request.path}",
                    ip_address=request.META.get('REMOTE_ADDR')
                )
            
            return response
        return _wrapped_view
    return decorator

def require_profile_complete(view_func):
    """
    Decorator to ensure user profile is complete.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        
        profile = request.user.profile
        required_fields = ['user_type', 'institution']
        
        for field in required_fields:
            if not getattr(profile, field):
                messages.warning(request, 'Please complete your profile to continue.')
                return redirect('core:profile')
        
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def admin_required(view_func):
    """
    Decorator to restrict access to admin users.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_staff:
            raise PermissionDenied("Admin access required.")
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def prevent_when_maintenance(view_func):
    """
    Decorator to prevent access during maintenance mode.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        from .models import SiteSettings
        settings = SiteSettings.objects.first()
        
        if settings and settings.maintenance_mode and not request.user.is_staff:
            return redirect('core:maintenance')
        
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def rate_limit(max_requests=60, timeout=60):
    """
    Decorator to rate limit views.
    Usage: @rate_limit(max_requests=10, timeout=60)
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            # Implement rate limiting logic here
            # This could use Redis or cache
            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator