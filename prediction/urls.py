"""
URL configuration for prediction app
"""

from django.urls import path
from . import views

app_name = 'prediction'

urlpatterns = [
    # QR Code Generation (ADDED)
    path('generate/', views.GenerateQRCodeView.as_view(), name='generate_qr'),
    path('qr-result/<int:pk>/', views.QRResultView.as_view(), name='view_qr_result'),
    
    # Static prediction
    path('static/', views.StaticPredictionView.as_view(), name='static_prediction'),
    path('static/result/<int:pk>/', views.PredictionResultView.as_view(), name='prediction_result'),
    path('report/<int:pk>/', views.PredictionPrintReportView.as_view(), name='print_report'),
    
    # Dynamic prediction (QR code)
    path('dynamic/', views.DynamicPredictionView.as_view(), name='dynamic_prediction'),
    path('dynamic/scan/', views.scan_qr_code, name='scan_qr_code'),
    path('dynamic/result/<str:data>/', views.dynamic_result, name='dynamic_result'),
    
    # Model management
    path('models/', views.ModelListView.as_view(), name='model_list'),
    path('models/train/', views.TrainModelView.as_view(), name='train_model'),
    path('models/<int:pk>/', views.ModelDetailView.as_view(), name='model_detail'),
    path('models/<int:pk>/update/', views.ModelUpdateView.as_view(), name='model_update'),
    path('models/<int:pk>/delete/', views.ModelDeleteView.as_view(), name='model_delete'),
    path('models/<int:pk>/download/', views.download_model, name='download_model'),
    path('models/<int:pk>/activate/', views.activate_model, name='activate_model'),
    path('models/<int:pk>/compare/', views.compare_model, name='compare_model'),
    
    # Batch prediction
    path('batch/', views.BatchPredictionView.as_view(), name='batch_upload'),
    path('batch/<int:pk>/', views.BatchDetailView.as_view(), name='batch_detail'),
    path('batch/<int:pk>/download/', views.download_batch_results, name='download_batch_results'),
    path('batch/<int:pk>/status/', views.batch_status, name='batch_status'),
    
    # Prediction history
    path('history/', views.PredictionHistoryView.as_view(), name='history'),
    path('history/<int:pk>/', views.HistoryDetailView.as_view(), name='history_detail'),
    path('history/<int:pk>/delete/', views.delete_prediction, name='delete_prediction'),
    path('history/<int:pk>/feedback/', views.add_feedback, name='add_feedback'),
    
    # Analytics
    path('analytics/', views.PredictionAnalyticsView.as_view(), name='analytics'),
    path('analytics/export/', views.export_analytics, name='export_analytics'),
    path('analytics/model-performance/', views.model_performance, name='model_performance'),
    
    # AJAX endpoints
    path('ajax/predict/', views.ajax_predict, name='ajax_predict'),
    path('ajax/training-status/<int:pk>/', views.training_status, name='training_status'),
    path('ajax/model-metrics/<int:pk>/', views.model_metrics, name='model_metrics'),
    path('ajax/prediction-stats/', views.prediction_stats, name='prediction_stats'),
    
    # API endpoints
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/models/', views.api_models, name='api_models'),
    path('api/models/<int:pk>/', views.api_model_detail, name='api_model_detail'),
]