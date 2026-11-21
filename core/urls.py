from django.urls import path
from django.http import HttpResponse
from . import views

app_name = 'core'

def index(request):
    return HttpResponse("Core App - Redirecting to home...")

def dashboard(request):
    return HttpResponse("""
    <h1>Dashboard</h1>
    <p>Welcome to your dashboard!</p>
    <a href="/">Back to Home</a>
    """)

urlpatterns = [
    path('', index, name='index'),
    path('dashboard/', dashboard, name='dashboard'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
]