from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import reverse_lazy
from django.http import FileResponse, Http404, JsonResponse, HttpResponse
from django.core.files.storage import default_storage
from django.views.decorators.http import require_POST
from django.template.loader import render_to_string
import json
import csv
import os
import mimetypes

from .models import UserProfile, ActivityLog, SiteSettings, Notification
from .forms import (
    UserRegistrationForm, UserLoginForm, UserProfileForm,
    UserUpdateForm, PasswordChangeCustomForm, AvatarUploadForm,
    ContactForm, SiteSettingsForm
)
from .decorators import admin_required
from .utils import (
    send_verification_email, send_contact_email, get_client_ip,
    create_activity_log, create_notification, calculate_storage_usage
)

# Remove or comment out this line:
# from weasyprint import HTML  # <-- REMOVE THIS LINE


def private_media(request, path):
    """Serve public demo media files after basic path validation."""
    if not path or '\\' in path or '..' in path.split('/'):
        raise Http404

    try:
        response = FileResponse(
            default_storage.open(path, 'rb'),
            content_type=mimetypes.guess_type(path)[0] or 'application/octet-stream',
        )
        response['Cache-Control'] = 'private, no-store'
        return response
    except FileNotFoundError as error:
        raise Http404 from error


class CustomLoginView(LoginView):
    """Custom login view"""
    form_class = UserLoginForm
    template_name = 'core/login.html'
    
    def form_valid(self, form):
        # Log the login activity
        user = form.get_user()
        create_activity_log(
            user=user,
            action='login',
            description=f"User logged in from {get_client_ip(self.request)}",
            request=self.request
        )
        
        # Update last login IP
        profile = user.profile
        profile.last_login_ip = get_client_ip(self.request)
        profile.save()
        
        # Create welcome notification
        create_notification(
            user=user,
            title='Welcome Back!',
            message=f'Welcome back to Alzheimer\'s Prediction System, {user.username}!',
            notification_type='success'
        )
        
        # Remember me functionality
        if not form.cleaned_data.get('remember_me'):
            self.request.session.set_expiry(0)
        
        return super().form_valid(form)
    
    def get_success_url(self):
        next_url = self.request.GET.get('next')
        if next_url:
            return next_url
        return reverse_lazy('core:dashboard')


class CustomLogoutView(LogoutView):
    """Custom logout view"""
    next_page = reverse_lazy('home')
    
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            create_activity_log(
                user=request.user,
                action='logout',
                description="User logged out",
                request=request
            )
        return super().dispatch(request, *args, **kwargs)


def register(request):
    """User registration view"""
    if request.user.is_authenticated:
        return redirect('core:dashboard')
    
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Send verification email
            send_verification_email(user, request)
            
            # Log registration
            create_activity_log(
                user=user,
                action='register',
                description=f"New user registered: {user.username}",
                request=request
            )
            
            messages.success(
                request, 
                'Registration successful! Please check your email to verify your account.'
            )
            return redirect('login')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'core/register.html', {'form': form})


def verify_email(request, token):
    """Verify user email"""
    try:
        profile = UserProfile.objects.get(verification_token=token)
        profile.is_verified = True
        profile.verification_token = None
        profile.save()
        
        messages.success(request, 'Email verified successfully! You can now login.')
    except UserProfile.DoesNotExist:
        messages.error(request, 'Invalid verification token.')
    
    return redirect('login')


@login_required
def dashboard(request):
    """User dashboard view"""
    from dataset_manager.models import Dataset
    from prediction.models import Prediction, TrainedModel
    
    # Get user statistics
    total_predictions = Prediction.objects.filter(created_by=request.user).count()
    total_datasets = Dataset.objects.filter(uploaded_by=request.user).count()
    total_models = TrainedModel.objects.filter(created_by=request.user).count()
    
    # Get recent predictions
    recent_predictions = Prediction.objects.filter(
        created_by=request.user
    ).order_by('-created_at')[:5]
    
    # Get recent datasets
    recent_datasets = Dataset.objects.filter(
        uploaded_by=request.user
    ).order_by('-created_at')[:5]
    
    # Get recent models
    recent_models = TrainedModel.objects.filter(
        created_by=request.user
    ).order_by('-created_at')[:5]
    
    # Get storage usage
    used_storage, limit_bytes = calculate_storage_usage(request.user)[1:3]
    storage_percentage = (used_storage / limit_bytes * 100) if limit_bytes > 0 else 0
    
    # Get activity logs
    recent_activities = ActivityLog.objects.filter(
        user=request.user
    ).order_by('-timestamp')[:10]
    
    # Get unread notifications count
    unread_notifications = Notification.objects.filter(
        user=request.user,
        is_read=False
    ).count()
    
    context = {
        'total_predictions': total_predictions,
        'total_datasets': total_datasets,
        'total_models': total_models,
        'recent_predictions': recent_predictions,
        'recent_datasets': recent_datasets,
        'recent_models': recent_models,
        'recent_activities': recent_activities,
        'storage_percentage': storage_percentage,
        'used_storage': used_storage,
        'unread_notifications': unread_notifications,
    }
    
    return render(request, 'core/dashboard.html', context)


@login_required
def profile(request):
    """User profile view"""
    return render(request, 'core/profile.html', {'user': request.user})


@login_required
def edit_profile(request):
    """Edit user profile"""
    if request.method == 'POST':
        user_form = UserUpdateForm(request.POST, instance=request.user)
        profile_form = UserProfileForm(request.POST, instance=request.user.profile)
        
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            
            create_activity_log(
                user=request.user,
                action='update',
                description="Profile updated",
                request=request
            )
            
            messages.success(request, 'Profile updated successfully!')
            return redirect('core:profile')
    else:
        user_form = UserUpdateForm(instance=request.user)
        profile_form = UserProfileForm(instance=request.user.profile)
    
    context = {
        'user_form': user_form,
        'profile_form': profile_form
    }
    return render(request, 'core/edit_profile.html', context)


@login_required
def change_password(request):
    """Change user password"""
    if request.method == 'POST':
        form = PasswordChangeCustomForm(request.POST)
        if form.is_valid():
            user = request.user
            if user.check_password(form.cleaned_data['old_password']):
                user.set_password(form.cleaned_data['new_password1'])
                user.save()
                update_session_auth_hash(request, user)
                
                create_activity_log(
                    user=request.user,
                    action='update',
                    description="Password changed",
                    request=request
                )
                
                messages.success(request, 'Password changed successfully!')
                return redirect('core:profile')
            else:
                messages.error(request, 'Current password is incorrect.')
    else:
        form = PasswordChangeCustomForm()
    
    return render(request, 'core/change_password.html', {'form': form})


@login_required
def upload_avatar(request):
    """Upload user avatar"""
    if request.method == 'POST':
        form = AvatarUploadForm(request.POST, request.FILES)
        if form.is_valid():
            avatar = form.cleaned_data['avatar']
            
            # Delete old avatar
            if request.user.profile.avatar:
                if os.path.isfile(request.user.profile.avatar.path):
                    os.remove(request.user.profile.avatar.path)
            
            request.user.profile.avatar = avatar
            request.user.profile.save()
            
            messages.success(request, 'Avatar uploaded successfully!')
            
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'avatar_url': request.user.profile.avatar.url
                })
    
    return redirect('core:profile')


@login_required
def activity_logs(request):
    """View user activity logs"""
    logs = ActivityLog.objects.filter(user=request.user).order_by('-timestamp')
    
    # Filter by action type
    action = request.GET.get('action')
    if action:
        logs = logs.filter(action=action)
    
    # Filter by date
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    if date_from:
        logs = logs.filter(timestamp__date__gte=date_from)
    if date_to:
        logs = logs.filter(timestamp__date__lte=date_to)
    
    # Pagination
    from django.core.paginator import Paginator
    paginator = Paginator(logs, 20)
    page = request.GET.get('page')
    logs = paginator.get_page(page)
    
    context = {
        'logs': logs,
        'action_choices': ActivityLog.ACTION_TYPES,
        'current_action': action,
        'date_from': date_from,
        'date_to': date_to,
    }
    return render(request, 'core/activity_logs.html', context)


@login_required
def notifications(request):
    """View user notifications"""
    notifications_list = Notification.objects.filter(user=request.user)
    
    # Filter by read status
    status = request.GET.get('status')
    if status == 'read':
        notifications_list = notifications_list.filter(is_read=True)
    elif status == 'unread':
        notifications_list = notifications_list.filter(is_read=False)
    
    # Pagination
    from django.core.paginator import Paginator
    paginator = Paginator(notifications_list, 20)
    page = request.GET.get('page')
    notifications = paginator.get_page(page)
    
    return render(request, 'core/notifications.html', {
        'notifications': notifications
    })


@login_required
@require_POST
def mark_notification_read(request, notification_id):
    """Mark notification as read"""
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.mark_as_read()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    return redirect('core:notifications')


@login_required
@require_POST
def mark_all_notifications_read(request):
    """Mark all notifications as read"""
    Notification.mark_all_read(request.user)
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    return redirect('core:notifications')


@admin_required
def site_settings(request):
    """Site settings view (admin only)"""
    settings_obj = SiteSettings.get_settings()
    
    if request.method == 'POST':
        form = SiteSettingsForm(request.POST, request.FILES, instance=settings_obj)
        if form.is_valid():
            form.save()
            messages.success(request, 'Settings updated successfully!')
            return redirect('core:settings')
    else:
        form = SiteSettingsForm(instance=settings_obj)
    
    return render(request, 'core/settings.html', {'form': form})


def about(request):
    """About page"""
    return render(request, 'core/about.html')


def contact(request):
    """Contact page"""
    if request.method == 'POST':
        form = ContactForm(request.POST, request.FILES)
        if form.is_valid():
            # Send email
            if send_contact_email(form.cleaned_data):
                messages.success(request, 'Message sent successfully! We\'ll get back to you soon.')
            else:
                messages.error(request, 'Failed to send message. Please try again later.')
            return redirect('core:contact')
    else:
        form = ContactForm()
    
    return render(request, 'core/contact.html', {'form': form})


def privacy_policy(request):
    """Privacy policy page"""
    return render(request, 'core/privacy_policy.html')


def terms_of_service(request):
    """Terms of service page"""
    return render(request, 'core/terms_of_service.html')


def faq(request):
    """FAQ page"""
    return render(request, 'core/faq.html')


def maintenance(request):
    """Maintenance mode page"""
    return render(request, 'core/maintenance.html')


@login_required
def api_documentation(request):
    """API documentation page"""
    return render(request, 'core/api_docs.html')


# Export functions
@login_required
def export_activity_logs(request):
    """Export activity logs as CSV"""
    logs = ActivityLog.objects.filter(user=request.user)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="activity_logs.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Action', 'Description', 'IP Address', 'Timestamp'])
    
    for log in logs:
        writer.writerow([log.action, log.description, log.ip_address, log.timestamp])
    
    return response


@login_required
def export_profile_pdf(request):
    """Export user profile as PDF"""
    # Comment out or remove this function if you don't need PDF export
    # Or install weasyprint: pip install weasyprint
    
    # Temporary simple response
    response = HttpResponse(content_type='text/html')
    response['Content-Disposition'] = 'attachment; filename="profile.html"'
    response.write(f"""
    <html>
    <head><title>Profile</title></head>
    <body>
        <h1>User Profile</h1>
        <p>Name: {request.user.get_full_name()}</p>
        <p>Email: {request.user.email}</p>
        <p>Username: {request.user.username}</p>
    </body>
    </html>
    """)
    return response