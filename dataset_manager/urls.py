from django.urls import path
from . import views

app_name = 'dataset_manager'

urlpatterns = [
    path('', views.DatasetListView.as_view(), name='list'),
    path('upload/', views.DatasetUploadView.as_view(), name='upload'),
    path('<int:pk>/', views.DatasetDetailView.as_view(), name='detail'),
]