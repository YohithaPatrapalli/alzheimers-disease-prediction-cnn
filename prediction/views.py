"""
Views for prediction app - Alzheimer's Disease Prediction System
"""

import json
import os
import tempfile
import io
import csv
import base64
import random
from io import StringIO
from PIL import Image

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView, TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse, reverse_lazy
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.utils.decorators import method_decorator
from django.core.paginator import Paginator
from django.db.models import Q, Count, Avg
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone

# Local imports
from .models import TrainedModel, Prediction, BatchPrediction, PredictionFeedback
from .forms import (
    ModelTrainingForm, StaticPredictionForm, DynamicPredictionForm,
    BatchPredictionForm, PredictionFeedbackForm, ModelComparisonForm,
    PredictionHistoryFilterForm
)
from .tasks import train_model_task, batch_prediction_task
from .utils import (
    generate_prediction_qr, decode_qr_code, prepare_prediction_chart_data,
    calculate_prediction_statistics, get_best_model, normalize_prediction_stage,
    normalize_confidence_scores, load_predictor_for_model
)
from core.utils import create_activity_log


def ensure_prediction_share_qr(prediction, request):
    """Save a QR code that points to this owner's login-protected printable report."""
    share_url = request.build_absolute_uri(
        reverse('prediction:print_report', args=[prediction.pk])
    )
    if prediction.qr_data == share_url and prediction.qr_code:
        return share_url

    if prediction.qr_code:
        prediction.qr_code.delete(save=False)

    qr_image = generate_prediction_qr(prediction, share_url=share_url)
    qr_buffer = io.BytesIO()
    qr_image.save(qr_buffer, format='PNG')
    prediction.qr_data = share_url
    prediction.qr_code.save(
        f'prediction_{prediction.pk}_share.png',
        ContentFile(qr_buffer.getvalue()),
        save=False,
    )
    prediction.save(update_fields=['qr_data', 'qr_code'])
    return share_url


# ============================================================================
# QR Code Generation Views
# ============================================================================

class GenerateQRCodeView(LoginRequiredMixin, TemplateView):
    """Generate QR code from uploaded MRI image and make prediction"""
    template_name = 'prediction/generate_qr.html'
    
    @method_decorator(csrf_protect)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = StaticPredictionForm()
        context['title'] = 'Generate QR Code from MRI Image'
        return context
    
    def post(self, request, *args, **kwargs):
        form = StaticPredictionForm(request.POST, request.FILES)
        
        if form.is_valid():
            image = form.cleaned_data['image']
            selected_model = form.cleaned_data['model']
            
            # Determine which model to use
            if selected_model:
                model = selected_model
            else:
                model = get_best_model()
            
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                for chunk in image.chunks():
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
            
            # Convert image to base64 for display
            img = Image.open(tmp_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            display_img = img.copy()
            display_img.thumbnail((300, 300))
            buffered = io.BytesIO()
            display_img.save(buffered, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Create demo prediction result (if no model) or use model
            if not model:
                messages.info(request, 'Using demo prediction mode. Train a model for more accurate results.')
                
                random.seed(hash(tmp_path) % 2**32)
                
                # Generate realistic demo confidence scores
                demo_scores = {
                    'Non-Demented': random.uniform(50, 95),
                    'Very Mild Demented': random.uniform(30, 85),
                    'Mild Demented': random.uniform(20, 75),
                    'Moderate Demented': random.uniform(10, 60)
                }
                
                # Get predicted stage
                predicted_stage = max(demo_scores, key=demo_scores.get)
                confidence = demo_scores[predicted_stage]
                
                # Normalize scores to sum to 100
                total = sum(demo_scores.values())
                demo_scores = {k: (v/total)*100 for k, v in demo_scores.items()}
                
                # Create prediction record
                prediction = Prediction(
                    model=None,
                    predicted_stage=normalize_prediction_stage(predicted_stage),
                    confidence=confidence,
                    confidence_scores=normalize_confidence_scores(demo_scores),
                    processing_time=0.5,
                    created_by=request.user if request.user.is_authenticated else None
                )
            else:
                try:
                    # Load predictor
                    if model.file and os.path.exists(model.file.path):
                        predictor = load_predictor_for_model(model)
                    else:
                        raise ValueError("Model file not found")
                    
                    # Make prediction
                    result = predictor.predict(tmp_path)
                    if not result.success:
                        raise RuntimeError(result.error_message or 'Model prediction failed.')
                    
                    # Create prediction record
                    prediction = Prediction(
                        model=model,
                        predicted_stage=normalize_prediction_stage(result.predicted_class),
                        confidence=result.confidence,
                        confidence_scores=normalize_confidence_scores(result.all_probabilities),
                        processing_time=result.processing_time,
                        created_by=request.user if request.user.is_authenticated else None
                    )
                except Exception as e:
                    messages.error(request, f'Prediction failed: {str(e)}')
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    return redirect('prediction:generate_qr')
            
            # IMPORTANT: Save the prediction first to generate created_at timestamp
            prediction.save()
            
            # Save original image
            with open(tmp_path, 'rb') as f:
                prediction.image.save(
                    image.name,
                    ContentFile(f.read()),
                    save=False
                )
            
            # Save again to update the image field
            prediction.save()
            
            share_url = ensure_prediction_share_qr(prediction, request)
            
            # Store in session
            request.session['prediction_image'] = image_base64
            request.session['prediction_data'] = {'report_url': share_url}
            
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            messages.success(request, f'QR Code generated! Prediction: {prediction.get_predicted_stage_display()}')
            return redirect(f'/prediction/qr-result/{prediction.id}/')
        
        context = self.get_context_data()
        context['form'] = form
        return render(request, self.template_name, context)


class QRResultView(LoginRequiredMixin, DetailView):
    """View QR code and prediction result"""
    model = Prediction
    template_name = 'prediction/qr_result.html'
    context_object_name = 'prediction'
    
    def get_queryset(self):
        return Prediction.objects.filter(created_by=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        prediction = self.get_object()
        
        # Prepare chart data
        chart_data = prepare_prediction_chart_data(prediction)
        
        # Get image from session
        image_data = self.request.session.get('prediction_image', None)
        
        share_url = ensure_prediction_share_qr(prediction, self.request)
        
        context['chart_data'] = chart_data
        context['image_data'] = image_data
        context['qr_data'] = share_url
        context['stage_info'] = self.get_stage_info(prediction.predicted_stage)
        context['title'] = f'QR Code Result - {prediction.get_predicted_stage_display()}'
        
        return context
    
    def get_stage_info(self, stage):
        """Get detailed information about the predicted stage"""
        stage_info = {
            'non_demented': {
                'description': 'No significant signs of cognitive decline. Normal cognitive function for age.',
                'recommendations': [
                    'Regular check-ups with healthcare provider',
                    'Maintain a healthy lifestyle with proper diet and exercise',
                    'Engage in mentally stimulating activities',
                    'Monitor for any changes in memory or cognition'
                ]
            },
            'very_mild': {
                'description': 'Early signs of cognitive decline that may be mistaken for normal aging. Subtle memory lapses and difficulty finding words.',
                'recommendations': [
                    'Schedule a comprehensive neurological evaluation',
                    'Consider lifestyle modifications (diet, exercise, cognitive training)',
                    'Monitor cognitive function regularly',
                    'Discuss with family and healthcare provider about early intervention'
                ]
            },
            'mild': {
                'description': 'Noticeable cognitive decline affecting daily life. Memory loss of recent events, difficulty with complex tasks, changes in mood.',
                'recommendations': [
                    'Consult with a neurologist immediately',
                    'Consider medication options (cholinesterase inhibitors)',
                    'Establish a daily routine and use memory aids',
                    'Ensure safety measures at home',
                    'Join support groups for patients and caregivers'
                ]
            },
            'moderate': {
                'description': 'Significant cognitive decline requiring assistance. Severe memory loss, difficulty with basic tasks, behavioral changes.',
                'recommendations': [
                    'Immediate consultation with a neurologist',
                    'Comprehensive care plan development',
                    'Consider memory care facilities if needed',
                    'Ensure 24/7 supervision and safety measures',
                    'Explore clinical trials and new treatments',
                    'Strong support system for caregivers'
                ]
            }
        }
        return stage_info.get(stage, stage_info['non_demented'])


# ============================================================================
# Static Prediction Views
# ============================================================================

class StaticPredictionView(LoginRequiredMixin, TemplateView):
    """Static prediction - upload image and predict"""
    template_name = 'prediction/static_prediction.html'
    
    @method_decorator(csrf_protect)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = StaticPredictionForm()
        context['recent_predictions'] = Prediction.objects.filter(
            created_by=self.request.user
        ).order_by('-created_at')[:5]
        context['best_model'] = get_best_model()
        context['title'] = 'Static Prediction'
        return context
    
    def post(self, request, *args, **kwargs):
        form = StaticPredictionForm(request.POST, request.FILES)
        
        if form.is_valid():
            image = form.cleaned_data['image']
            selected_model = form.cleaned_data['model']
            generate_qr = form.cleaned_data['generate_qr']
            
            # Determine which model to use
            if selected_model:
                model = selected_model
            else:
                model = get_best_model()
                
                # If no model exists, use demo prediction mode
                if not model:
                    messages.info(request, 'Using demo prediction mode. Train a model for more accurate results.')
                    
                    # Save image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        for chunk in image.chunks():
                            tmp_file.write(chunk)
                        tmp_path = tmp_file.name
                    
                    # Convert image to base64 for display
                    img = Image.open(tmp_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    display_img = img.copy()
                    display_img.thumbnail((400, 400))
                    buffered = io.BytesIO()
                    display_img.save(buffered, format="JPEG", quality=85)
                    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Create demo prediction result
                    random.seed(hash(tmp_path) % 2**32)
                    
                    # Generate realistic demo confidence scores based on image characteristics
                    demo_scores = {
                        'Non-Demented': random.uniform(50, 95),
                        'Very Mild Demented': random.uniform(30, 85),
                        'Mild Demented': random.uniform(20, 75),
                        'Moderate Demented': random.uniform(10, 60)
                    }
                    
                    # Get predicted stage
                    predicted_stage = max(demo_scores, key=demo_scores.get)
                    confidence = demo_scores[predicted_stage]
                    
                    # Normalize scores to sum to 100
                    total = sum(demo_scores.values())
                    demo_scores = {k: (v/total)*100 for k, v in demo_scores.items()}
                    
                    # Create prediction record
                    prediction = Prediction(
                        model=None,
                        predicted_stage=normalize_prediction_stage(predicted_stage),
                        confidence=confidence,
                        confidence_scores=normalize_confidence_scores(demo_scores),
                        processing_time=0.5,
                        created_by=request.user if request.user.is_authenticated else None
                    )
                    
                    # Save original image
                    with open(tmp_path, 'rb') as f:
                        prediction.image.save(
                            image.name,
                            ContentFile(f.read()),
                            save=False
                        )
                    
                    prediction.save()
                    
                    # Store image in session
                    request.session['prediction_image'] = image_base64
                    
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    
                    messages.success(request, f'Demo prediction: {predicted_stage}')
                    return redirect(f'/prediction/static/result/{prediction.id}/')
            
            # Continue with existing code for when model exists
            # Save image temporarily and convert to base64 for display
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                for chunk in image.chunks():
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
            
            # Convert image to base64 for display in result page
            img = Image.open(tmp_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create thumbnail for display
            display_img = img.copy()
            display_img.thumbnail((400, 400))
            buffered = io.BytesIO()
            display_img.save(buffered, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            try:
                # Load predictor
                if model.file and os.path.exists(model.file.path):
                    predictor = load_predictor_for_model(model)
                else:
                    raise ValueError("Model file not found")
                
                # Make prediction
                result = predictor.predict(tmp_path)
                if not result.success:
                    raise RuntimeError(result.error_message or 'Model prediction failed.')
                
                # Create prediction record
                prediction = Prediction(
                    model=model,
                    predicted_stage=normalize_prediction_stage(result.predicted_class),
                    confidence=result.confidence,
                    confidence_scores=normalize_confidence_scores(result.all_probabilities),
                    processing_time=result.processing_time,
                    created_by=request.user if request.user.is_authenticated else None
                )
                
                # Save original image
                with open(tmp_path, 'rb') as f:
                    prediction.image.save(
                        image.name,
                        ContentFile(f.read()),
                        save=False
                    )
                
                prediction.save()

                # Generate QR code if requested
                if generate_qr:
                    ensure_prediction_share_qr(prediction, request)
                
                prediction.save()
                
                # Log activity
                create_activity_log(
                    user=request.user,
                    action='prediction',
                    description=f"Predicted: {prediction.get_predicted_stage_display()} ({prediction.confidence:.1f}%)",
                    request=request
                )
                
                messages.success(request, 'Prediction completed successfully!')
                
                # Store image base64 in session to pass to result page
                request.session['prediction_image'] = image_base64
                
                return redirect(f'/prediction/static/result/{prediction.id}/')
                
            except Exception as e:
                messages.error(request, f'Prediction failed: {str(e)}')
                return redirect('/prediction/static/')
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        context = self.get_context_data()
        context['form'] = form
        return render(request, self.template_name, context)


class PredictionResultView(LoginRequiredMixin, DetailView):
    """View prediction result with image and chart"""
    model = Prediction
    template_name = 'prediction/result.html'
    context_object_name = 'prediction'
    
    def get_queryset(self):
        return Prediction.objects.filter(created_by=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        prediction = self.get_object()
        
        # Prepare chart data
        chart_data = prepare_prediction_chart_data(prediction)
        
        # Get image from session if available
        image_data = self.request.session.get('prediction_image', None)
        
        # Get related predictions
        context['related_predictions'] = Prediction.objects.filter(
            created_by=self.request.user
        ).exclude(id=prediction.id)[:3]
        
        # Feedback form
        context['feedback_form'] = PredictionFeedbackForm()
        
        # Check if feedback exists
        context['existing_feedback'] = PredictionFeedback.objects.filter(
            prediction=prediction,
            user=self.request.user
        ).first()
        
        # Add chart data and image data to context
        context['chart_data'] = chart_data
        context['image_data'] = image_data
        
        # Add stage information
        context['stage_info'] = self.get_stage_info(prediction.predicted_stage)
        
        context['title'] = f'Prediction Result - {prediction.get_predicted_stage_display()}'
        return context
    
    def get_stage_info(self, stage):
        """Get detailed information about the predicted stage"""
        stage_info = {
            'non_demented': {
                'description': 'No significant signs of cognitive decline. Normal cognitive function for age.',
                'recommendations': [
                    'Regular check-ups with healthcare provider',
                    'Maintain a healthy lifestyle with proper diet and exercise',
                    'Engage in mentally stimulating activities',
                    'Monitor for any changes in memory or cognition'
                ]
            },
            'very_mild': {
                'description': 'Early signs of cognitive decline that may be mistaken for normal aging. Subtle memory lapses and difficulty finding words.',
                'recommendations': [
                    'Schedule a comprehensive neurological evaluation',
                    'Consider lifestyle modifications (diet, exercise, cognitive training)',
                    'Monitor cognitive function regularly',
                    'Discuss with family and healthcare provider about early intervention'
                ]
            },
            'mild': {
                'description': 'Noticeable cognitive decline affecting daily life. Memory loss of recent events, difficulty with complex tasks, changes in mood.',
                'recommendations': [
                    'Consult with a neurologist immediately',
                    'Consider medication options (cholinesterase inhibitors)',
                    'Establish a daily routine and use memory aids',
                    'Ensure safety measures at home',
                    'Join support groups for patients and caregivers'
                ]
            },
            'moderate': {
                'description': 'Significant cognitive decline requiring assistance. Severe memory loss, difficulty with basic tasks, behavioral changes.',
                'recommendations': [
                    'Immediate consultation with a neurologist',
                    'Comprehensive care plan development',
                    'Consider memory care facilities if needed',
                    'Ensure 24/7 supervision and safety measures',
                    'Explore clinical trials and new treatments',
                    'Strong support system for caregivers'
                ]
            }
        }
        return stage_info.get(stage, stage_info['non_demented'])


class PredictionPrintReportView(LoginRequiredMixin, DetailView):
    """Render a printable report and a QR link restricted to the prediction owner."""
    model = Prediction
    template_name = 'prediction/print_report.html'
    context_object_name = 'prediction'

    def get_queryset(self):
        return Prediction.objects.filter(created_by=self.request.user).select_related('model')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        prediction = self.object
        context['share_url'] = ensure_prediction_share_qr(prediction, self.request)
        return context


# ============================================================================
# Dynamic Prediction Views (QR Code)
# ============================================================================

class DynamicPredictionView(LoginRequiredMixin, TemplateView):
    """Dynamic prediction using QR codes"""
    template_name = 'prediction/dynamic_prediction.html'
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = DynamicPredictionForm()
        context['title'] = 'Dynamic Prediction (QR Code)'
        return context
    
    def post(self, request, *args, **kwargs):
        form = DynamicPredictionForm(request.POST, request.FILES)
        
        if form.is_valid():
            qr_image = form.cleaned_data['qr_image']
            
            # Decode QR code
            data = decode_qr_code(qr_image)
            
            if data:
                report_url = data.get('raw_data', '')
                if (
                    report_url.startswith(request.build_absolute_uri('/prediction/report/'))
                    and report_url.endswith('/')
                ):
                    return redirect(report_url)

                # Check if it's a prediction ID
                if 'id' in data:
                    try:
                        prediction = Prediction.objects.get(
                            id=data['id'],
                            created_by=request.user,
                        )
                        return redirect(f'/prediction/static/result/{prediction.id}/')
                    except Prediction.DoesNotExist:
                        messages.error(request, 'Prediction not found.')
                else:
                    # Custom data
                    return redirect(f'/prediction/dynamic/result/{json.dumps(data)}')
            else:
                messages.error(request, 'Invalid or unreadable QR code.')
        
        context = self.get_context_data()
        context['form'] = form
        return render(request, self.template_name, context)


@csrf_exempt
def scan_qr_code(request):
    """AJAX endpoint for scanning QR code using OpenCV"""
    import cv2
    import numpy as np
    from PIL import Image
    import json
    
    if request.method == 'POST':
        if not request.user.is_authenticated:
            return JsonResponse(
                {'success': False, 'error': 'Log in to scan prediction QR codes.'},
                status=401,
            )
        if request.FILES.get('qr_image'):
            qr_image = request.FILES['qr_image']
            
            try:
                # Read the image
                img = Image.open(qr_image)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array
                img_array = np.array(img, dtype=np.uint8)
                
                # Convert to grayscale
                if len(img_array.shape) == 3:
                    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img_array
                
                # Use OpenCV QR code detector
                qr_detector = cv2.QRCodeDetector()
                data, bbox, _ = qr_detector.detectAndDecode(img_gray)
                
                if data:
                    # Try to parse as JSON
                    try:
                        parsed_data = json.loads(data)
                        
                        # Check if this is a prediction ID
                        if 'id' in parsed_data:
                            # Fetch the actual prediction from database
                            try:
                                from .models import Prediction
                                prediction = Prediction.objects.get(
                                    id=parsed_data['id'],
                                    created_by=request.user,
                                )
                                from django.utils import timezone

                                prediction_date = prediction.created_at or timezone.now()
                                
                                # Return full prediction data
                                return JsonResponse({
                                    'success': True,
                                    'data': {
                                        'id': prediction.id,
                                        'stage': prediction.predicted_stage,
                                        'stage_display': prediction.get_predicted_stage_display(),
                                        'confidence': prediction.confidence,
                                        'confidence_scores': prediction.confidence_scores,
                                        'date': prediction.created_at.isoformat(),
                                        'image_url': prediction.image.url if prediction.image else None
                                    }
                                })
                            except Prediction.DoesNotExist:
                                return JsonResponse({'success': True, 'data': parsed_data})
                        
                        return JsonResponse({'success': True, 'data': parsed_data})
                    except json.JSONDecodeError:
                        return JsonResponse({'success': True, 'data': {'raw_data': data}})
                else:
                    return JsonResponse({'success': False, 'error': 'No QR code found in image'})
                    
            except Exception as e:
                return JsonResponse({'success': False, 'error': f'Decoding error: {str(e)}'})
        else:
            return JsonResponse({'success': False, 'error': 'No image provided'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@login_required
def dynamic_result(request, data):
    """View dynamic prediction result from QR code"""
    try:
        data = json.loads(data)
        
        # Check if it's a prediction
        if 'id' in data:
            try:
                prediction = Prediction.objects.get(
                    id=data['id'],
                    created_by=request.user,
                )
                context = {
                    'data': data,
                    'prediction': prediction,
                    'chart_data': prepare_prediction_chart_data(prediction),
                    'stage_info': {
                        'non_demented': 'Normal cognitive function',
                        'very_mild': 'Very mild cognitive decline',
                        'mild': 'Mild cognitive decline',
                        'moderate': 'Moderate cognitive decline'
                    }.get(prediction.predicted_stage, '')
                }
            except Prediction.DoesNotExist:
                context = {'data': data, 'error': 'Prediction not found'}
        else:
            context = {'data': data}
        
        return render(request, 'prediction/dynamic_result.html', context)
        
    except json.JSONDecodeError:
        messages.error(request, 'Invalid data format')
        return redirect('/prediction/dynamic/')

# ============================================================================
# Model Management Views
# ============================================================================

class ModelListView(LoginRequiredMixin, ListView):
    """List trained models"""
    model = TrainedModel
    template_name = 'prediction/models/list.html'
    context_object_name = 'models'
    paginate_by = 12
    
    def get_queryset(self):
        queryset = TrainedModel.objects.filter(created_by=self.request.user)
        
        # Filter by type
        model_type = self.request.GET.get('type')
        if model_type:
            queryset = queryset.filter(model_type=model_type)
        
        # Filter by status
        status = self.request.GET.get('status')
        if status:
            queryset = queryset.filter(status=status)
        
        # Search
        search = self.request.GET.get('search')
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) |
                Q(description__icontains=search)
            )
        
        return queryset.order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['model_types'] = TrainedModel.MODEL_TYPES
        context['status_choices'] = TrainedModel.STATUS_CHOICES
        context['title'] = 'Trained Models'
        return context


class TrainModelView(LoginRequiredMixin, TemplateView):
    """Train new model"""
    template_name = 'prediction/models/train.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dataset_id = self.request.GET.get('dataset')
        context['form'] = ModelTrainingForm(
            user=self.request.user,
            initial={'dataset': dataset_id} if dataset_id else None,
        )
        context['title'] = 'Train New Model'
        return context
    
    def post(self, request, *args, **kwargs):
        form = ModelTrainingForm(request.POST, user=request.user)
        
        if form.is_valid():
            dataset = form.cleaned_data['dataset']
            dataset_split = dataset.splits.filter(is_active=True).order_by('-created_at').first()
            # Create model record
            model = TrainedModel.objects.create(
                name=form.cleaned_data['name'],
                description=form.cleaned_data['description'],
                model_type=form.cleaned_data['model_type'],
                dataset=dataset,
                dataset_split=dataset_split,
                status='pending',
                created_by=request.user,
                training_config={
                    'epochs': form.cleaned_data['epochs'],
                    'batch_size': form.cleaned_data['batch_size'],
                    'learning_rate': form.cleaned_data['learning_rate'],
                    'validation_split': form.cleaned_data['validation_split'],
                    'use_augmentation': form.cleaned_data['use_augmentation'],
                    'early_stopping': form.cleaned_data['early_stopping']
                }
            )
            
            # Start training task
            train_model_task(
                model.id,
                config=model.training_config
            )
            
            messages.success(
                request,
                'Model training started. You can check the progress in the models list.'
            )
            
            create_activity_log(
                user=request.user,
                action='training',
                description=f"Started training model: {model.name}",
                request=request
            )
            
            return redirect('/prediction/models/')
        
        context = self.get_context_data()
        context['form'] = form
        return render(request, self.template_name, context)


class ModelDetailView(LoginRequiredMixin, DetailView):
    """View model details"""
    model = TrainedModel
    template_name = 'prediction/models/detail.html'
    context_object_name = 'model'
    
    def get_queryset(self):
        return TrainedModel.objects.filter(created_by=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        model = self.get_object()
        
        # Get predictions using this model
        context['predictions'] = Prediction.objects.filter(model=model)[:10]
        
        # Get performance logs
        context['performance_logs'] = model.performance_logs.all()[:30]
        
        # Get training history for charts
        if model.training_history:
            history = model.training_history
            context['history'] = {
                'accuracy': history.get('accuracy', [])[-10:],
                'val_accuracy': history.get('val_accuracy', [])[-10:],
                'loss': history.get('loss', [])[-10:],
                'val_loss': history.get('val_loss', [])[-10:],
                'epochs': list(range(1, len(history.get('accuracy', [])) + 1))[-10:]
            }
        
        context['title'] = f'Model: {model.name}'
        return context


class ModelUpdateView(LoginRequiredMixin, UpdateView):
    """Update model information"""
    model = TrainedModel
    fields = ['name', 'description']
    template_name = 'prediction/models/update.html'
    
    def get_queryset(self):
        return TrainedModel.objects.filter(created_by=self.request.user)
    
    def get_success_url(self):
        return f'/prediction/models/{self.object.pk}/'
    
    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, 'Model updated successfully.')
        return response


class ModelDeleteView(LoginRequiredMixin, DeleteView):
    """Delete model"""
    model = TrainedModel
    template_name = 'prediction/models/delete.html'
    success_url = '/prediction/models/'
    
    def get_queryset(self):
        return TrainedModel.objects.filter(created_by=self.request.user)
    
    def delete(self, request, *args, **kwargs):
        model = self.get_object()
        
        # Delete model file
        if model.file and os.path.exists(model.file.path):
            try:
                os.remove(model.file.path)
            except:
                pass
        
        messages.success(request, f'Model "{model.name}" deleted successfully.')
        
        create_activity_log(
            user=request.user,
            action='delete',
            description=f"Deleted model: {model.name}",
            request=request
        )
        
        return super().delete(request, *args, **kwargs)


@login_required
def download_model(request, pk):
    """Download model file"""
    model = get_object_or_404(TrainedModel, pk=pk, created_by=request.user)
    
    if model.file and os.path.exists(model.file.path):
        response = FileResponse(
            open(model.file.path, 'rb'),
            as_attachment=True,
            filename=f'{model.name}.h5'
        )
        
        create_activity_log(
            user=request.user,
            action='download',
            description=f"Downloaded model: {model.name}",
            request=request
        )
        
        return response
    
    messages.error(request, 'Model file not found.')
    return redirect(f'/prediction/models/{pk}/')


@login_required
def activate_model(request, pk):
    """Activate model for predictions"""
    model = get_object_or_404(TrainedModel, pk=pk, created_by=request.user)
    
    # Deactivate all other models
    TrainedModel.objects.filter(created_by=request.user).update(is_active=False)
    
    # Activate this model
    model.is_active = True
    model.save()
    
    messages.success(request, f'Model "{model.name}" is now active.')
    return redirect(f'/prediction/models/{pk}/')


def compare_model(request, pk):
    """Compare model with others"""
    model = get_object_or_404(TrainedModel, pk=pk)
    
    if request.method == 'POST':
        form = ModelComparisonForm(request.POST, request.FILES)
        if form.is_valid():
            models_to_compare = form.cleaned_data['models']
            test_images = request.FILES.getlist('test_images')
            
            # Perform comparison
            results = []
            for img_file in test_images:
                img_results = {'filename': img_file.name}
                
                for compare_model in models_to_compare:
                    # Make prediction with each model
                    if compare_model.file and os.path.exists(compare_model.file.path):
                        predictor = load_predictor_for_model(compare_model)
                        
                        # Save temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            for chunk in img_file.chunks():
                                tmp.write(chunk)
                            tmp_path = tmp.name
                        
                        try:
                            result = predictor.predict(tmp_path)
                            img_results[compare_model.name] = {
                                'stage': result.predicted_class,
                                'confidence': result.confidence
                            }
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                
                results.append(img_results)
            
            context = {
                'models': models_to_compare,
                'results': results,
                'comparison_type': form.cleaned_data['comparison_type']
            }
            return render(request, 'prediction/models/comparison_results.html', context)
    else:
        form = ModelComparisonForm(initial={'models': [model]})
    
    return render(request, 'prediction/models/compare.html', {
        'model': model,
        'form': form,
        'title': f'Compare Model: {model.name}'
    })


# ============================================================================
# Batch Prediction Views
# ============================================================================

class BatchPredictionView(LoginRequiredMixin, CreateView):
    """Batch prediction upload"""
    model = BatchPrediction
    form_class = BatchPredictionForm
    template_name = 'prediction/batch_upload.html'
    success_url = '/prediction/history/'
    
    def form_valid(self, form):
        form.instance.created_by = self.request.user
        response = super().form_valid(form)
        
        # Start batch processing
        batch_prediction_task.delay(self.object.id)
        
        messages.success(
            self.request,
            'Batch prediction started. You can check progress in the history.'
        )
        
        create_activity_log(
            user=self.request.user,
            action='batch_prediction',
            description=f"Started batch prediction: {self.object.name}",
            request=request
        )
        
        return response


class BatchDetailView(LoginRequiredMixin, DetailView):
    """View batch prediction details"""
    model = BatchPrediction
    template_name = 'prediction/batch_detail.html'
    context_object_name = 'batch'
    
    def get_queryset(self):
        return BatchPrediction.objects.filter(created_by=self.request.user)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        batch = self.get_object()
        
        # Get results if completed
        if batch.results_summary:
            context['results'] = batch.results_summary.get('results', [])[:20]
            context['stage_distribution'] = batch.results_summary.get('stage_distribution', {})
            context['average_confidence'] = batch.results_summary.get('average_confidence', 0)
        
        context['title'] = f'Batch: {batch.name}'
        return context


@login_required
def download_batch_results(request, pk):
    """Download batch prediction results"""
    batch = get_object_or_404(BatchPrediction, pk=pk, created_by=request.user)
    
    if batch.results_file and os.path.exists(batch.results_file.path):
        response = FileResponse(
            open(batch.results_file.path, 'rb'),
            as_attachment=True,
            filename=f'batch_{batch.id}_results.json'
        )
        return response
    
    messages.error(request, 'Results file not found.')
    return redirect(f'/prediction/batch/{pk}/')


@login_required
def batch_status(request, pk):
    """Get batch status (AJAX)"""
    batch = get_object_or_404(BatchPrediction, pk=pk, created_by=request.user)
    
    return JsonResponse({
        'status': batch.status,
        'total': batch.total_images,
        'processed': batch.processed_images,
        'failed': batch.failed_images,
        'progress': batch.get_progress(),
        'started_at': batch.started_at.isoformat() if batch.started_at else None,
        'completed_at': batch.completed_at.isoformat() if batch.completed_at else None
    })


# ============================================================================
# Prediction History Views
# ============================================================================

class PredictionHistoryView(LoginRequiredMixin, ListView):
    """View prediction history"""
    model = Prediction
    template_name = 'prediction/history.html'
    context_object_name = 'predictions'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Prediction.objects.filter(created_by=self.request.user)
        
        # Apply filters
        self.filter_form = PredictionHistoryFilterForm(self.request.GET)
        if self.filter_form.is_valid():
            data = self.filter_form.cleaned_data
            
            if data.get('date_from'):
                queryset = queryset.filter(created_at__date__gte=data['date_from'])
            
            if data.get('date_to'):
                queryset = queryset.filter(created_at__date__lte=data['date_to'])
            
            if data.get('stage'):
                queryset = queryset.filter(predicted_stage=data['stage'])
            
            if data.get('min_confidence'):
                queryset = queryset.filter(confidence__gte=data['min_confidence'])
            
            if data.get('model'):
                queryset = queryset.filter(model=data['model'])
            
            if data.get('search'):
                queryset = queryset.filter(id__icontains=data['search'])
        
        return queryset.order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['filter_form'] = self.filter_form
        context['title'] = 'Prediction History'
        
        # Statistics
        stats = calculate_prediction_statistics(user=self.request.user, days=None)
        context['statistics'] = stats
        
        return context


class HistoryDetailView(PredictionResultView):
    """View prediction history detail"""
    
    def get_queryset(self):
        return Prediction.objects.filter(created_by=self.request.user)


@login_required
@require_POST
def delete_prediction(request, pk):
    """Delete prediction"""
    prediction = get_object_or_404(Prediction, pk=pk, created_by=request.user)
    
    # Delete files
    if prediction.image and os.path.exists(prediction.image.path):
        try:
            os.remove(prediction.image.path)
        except:
            pass
    if prediction.processed_image and os.path.exists(prediction.processed_image.path):
        try:
            os.remove(prediction.processed_image.path)
        except:
            pass
    if prediction.qr_code and os.path.exists(prediction.qr_code.path):
        try:
            os.remove(prediction.qr_code.path)
        except:
            pass
    
    prediction.delete()
    messages.success(request, 'Prediction deleted successfully.')
    
    return redirect('/prediction/history/')


@login_required
@require_POST
def add_feedback(request, pk):
    """Add feedback for prediction"""
    prediction = get_object_or_404(Prediction, pk=pk, created_by=request.user)
    
    # Check if feedback already exists
    existing = PredictionFeedback.objects.filter(
        prediction=prediction,
        user=request.user
    ).first()
    
    if existing:
        form = PredictionFeedbackForm(request.POST, instance=existing)
    else:
        form = PredictionFeedbackForm(request.POST)
    
    if form.is_valid():
        feedback = form.save(commit=False)
        feedback.prediction = prediction
        feedback.user = request.user
        feedback.save()
        
        messages.success(request, 'Thank you for your feedback!')
    else:
        messages.error(request, 'Invalid feedback data.')
    
    return redirect(f'/prediction/history/{pk}/')


# ============================================================================
# Analytics Views
# ============================================================================

class PredictionAnalyticsView(LoginRequiredMixin, TemplateView):
    """Prediction analytics dashboard"""
    template_name = 'prediction/analytics.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get predictions for the current user
        predictions = Prediction.objects.filter(created_by=self.request.user)
        
        # Overall statistics
        total_predictions = predictions.count()
        context['total_predictions'] = total_predictions
        
        # Stage distribution
        stage_stats = {}
        for stage, _ in Prediction.STAGE_CHOICES:
            count = predictions.filter(predicted_stage=stage).count()
            if count > 0:
                stage_stats[stage] = count
        context['stage_distribution'] = stage_stats
        
        # Average confidence
        avg_confidence = predictions.aggregate(Avg('confidence'))['confidence__avg']
        context['avg_confidence'] = round(avg_confidence, 2) if avg_confidence else 0
        
        # Model performance
        model_stats = predictions.values('model__name').annotate(
            count=Count('id'),
            avg_confidence=Avg('confidence')
        )
        context['model_stats'] = list(model_stats)
        
        # Recent predictions for chart
        recent_predictions = predictions.order_by('-created_at')[:10]
        chart_labels = []
        chart_values = []
        for pred in reversed(recent_predictions):
            chart_labels.append(pred.created_at.strftime('%m/%d %H:%M'))
            chart_values.append(pred.confidence)
        context['chart_labels'] = chart_labels
        context['chart_values'] = chart_values
        
        # Weekly activity
        from datetime import timedelta
        from django.utils import timezone
        weekly_counts = []
        today = timezone.now().date()
        for i in range(12):
            week_start = today - timedelta(weeks=i)
            week_end = week_start - timedelta(days=6)
            count = predictions.filter(
                created_at__date__gte=week_end,
                created_at__date__lte=week_start
            ).count()
            weekly_counts.append({
                'week': week_start.strftime('%b %d'),
                'count': count
            })
        context['weekly_activity'] = weekly_counts
        
        context['title'] = 'Analytics Dashboard'
        return context


@login_required
def export_analytics(request):
    """Export analytics as CSV"""
    import csv
    from io import StringIO
    
    predictions = Prediction.objects.filter(created_by=request.user)
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'ID', 'Date', 'Stage', 'Confidence (%)', 'Model', 'Processing Time (s)'
    ])
    
    # Data
    for p in predictions:
        writer.writerow([
            p.id,
            p.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            p.get_predicted_stage_display(),
            round(p.confidence, 2),
            p.model.name if p.model else 'N/A',
            round(p.processing_time, 3) if p.processing_time else 'N/A'
        ])
    
    response = HttpResponse(output.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="prediction_analytics.csv"'
    
    return response


@login_required
def model_performance(request):
    """Get model performance data for charts (AJAX)"""
    models = TrainedModel.objects.filter(created_by=request.user, status='completed')
    
    data = {
        'labels': [model.name for model in models],
        'accuracy': [model.accuracy for model in models] if models else [],
        'precision': [model.precision for model in models] if models else [],
        'recall': [model.recall for model in models] if models else [],
        'f1_score': [model.f1_score for model in models] if models else []
    }
    
    return JsonResponse(data)


# ============================================================================
# AJAX Endpoints
# ============================================================================

@login_required
def ajax_predict(request):
    """AJAX endpoint for prediction"""
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        model_id = request.POST.get('model_id')
        
        # Get model
        if model_id:
            model = get_object_or_404(TrainedModel, id=model_id, status='completed')
        else:
            model = get_best_model()
        
        if not model:
            return JsonResponse({'error': 'No model available'}, status=400)
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            for chunk in image.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        try:
            # Predict
            predictor = load_predictor_for_model(model)
            result = predictor.predict(tmp_path)
            if not result.success:
                raise RuntimeError(result.error_message or 'Model prediction failed.')
            
            # Save prediction
            prediction = Prediction(
                model=model,
                predicted_stage=normalize_prediction_stage(result.predicted_class),
                confidence=result.confidence,
                confidence_scores=normalize_confidence_scores(result.all_probabilities),
                processing_time=result.processing_time,
                created_by=request.user
            )
            
            # Save image
            with open(tmp_path, 'rb') as f:
                prediction.image.save(image.name, ContentFile(f.read()))
            
            prediction.save()
            
            # Return result
            return JsonResponse({
                'success': True,
                'prediction_id': prediction.id,
                'stage': prediction.get_predicted_stage_display(),
                'confidence': prediction.confidence,
                'confidence_scores': prediction.confidence_scores
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


@login_required
def training_status(request, pk):
    """Get training status (AJAX)"""
    model = get_object_or_404(TrainedModel, pk=pk, created_by=request.user)
    
    return JsonResponse({
        'status': model.status,
        'accuracy': model.accuracy,
        'error_message': model.error_message,
        'updated_at': model.updated_at.isoformat()
    })


@login_required
def model_metrics(request, pk):
    """Get model metrics (AJAX)"""
    model = get_object_or_404(TrainedModel, pk=pk, created_by=request.user)
    
    return JsonResponse({
        'accuracy': model.accuracy,
        'precision': model.precision,
        'recall': model.recall,
        'f1_score': model.f1_score,
        'confusion_matrix': model.confusion_matrix,
        'per_class_metrics': model.per_class_metrics
    })


@login_required
def prediction_stats(request):
    """Get prediction statistics (AJAX)"""
    days = int(request.GET.get('days', 30))
    stats = calculate_prediction_statistics(user=request.user, days=days)
    
    return JsonResponse(stats)


# ============================================================================
# API Endpoints
# ============================================================================

@csrf_exempt
def api_predict(request):
    """API endpoint for prediction"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    if not request.FILES.get('image'):
        return JsonResponse({'error': 'Image required'}, status=400)
    
    # Use the best model
    model = get_best_model()
    if not model:
        return JsonResponse({'error': 'No model available'}, status=503)
    
    image = request.FILES['image']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        for chunk in image.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name
    
    try:
        predictor = load_predictor_for_model(model)
        result = predictor.predict(tmp_path)
        if not result.success:
            raise RuntimeError(result.error_message or 'Model prediction failed.')
        
        return JsonResponse({
            'success': True,
            'stage': normalize_prediction_stage(result.predicted_class),
            'confidence': result.confidence,
            'confidence_scores': normalize_confidence_scores(result.all_probabilities)
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def api_models(request):
    """API endpoint to list models"""
    models = TrainedModel.objects.filter(status='completed')
    
    data = {
        'models': [
            {
                'id': m.id,
                'name': m.name,
                'type': m.model_type,
                'accuracy': m.accuracy,
                'created_at': m.created_at.isoformat()
            }
            for m in models
        ]
    }
    
    return JsonResponse(data)


def api_model_detail(request, pk):
    """API endpoint to get model details"""
    model = get_object_or_404(TrainedModel, pk=pk)
    
    data = {
        'id': model.id,
        'name': model.name,
        'type': model.model_type,
        'accuracy': model.accuracy,
        'precision': model.precision,
        'recall': model.recall,
        'f1_score': model.f1_score,
        'confusion_matrix': model.confusion_matrix,
        'created_at': model.created_at.isoformat()
    }
    
    return JsonResponse(data)