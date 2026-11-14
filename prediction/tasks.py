import os
import zipfile
import time
import json
import subprocess
import sys
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from django.core.files.base import ContentFile
from django.conf import settings
from django.utils import timezone
import logging

from .models import TrainedModel, Prediction, BatchPrediction, ModelPerformanceLog
from dataset_manager.models import DatasetSplit
from .utils import load_predictor_for_model, normalize_prediction_stage

logger = logging.getLogger(__name__)


def train_model_task(model_id, config=None):
    """Launch training in a separate process so web-server reloads do not stop it."""
    model = TrainedModel.objects.get(pk=model_id)
    if config is not None:
        model.training_config = config
    model.status = 'pending'
    model.error_message = None
    model.save(update_fields=['training_config', 'status', 'error_message', 'updated_at'])

    log_directory = os.path.join(settings.MEDIA_ROOT, 'logs', 'training')
    os.makedirs(log_directory, exist_ok=True)
    log_path = os.path.join(log_directory, f'model_{model_id}.log')
    command = [
        sys.executable,
        str(settings.BASE_DIR / 'manage.py'),
        'train_model_background',
        str(model_id),
    ]
    process_options = {
        'stdin': subprocess.DEVNULL,
        'stderr': subprocess.STDOUT,
        'cwd': str(settings.BASE_DIR),
        'close_fds': True,
    }
    if os.name == 'nt':
        process_options['creationflags'] = (
            getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x00000200)
            | getattr(subprocess, 'DETACHED_PROCESS', 0x00000008)
        )
    else:
        process_options['start_new_session'] = True

    try:
        with open(log_path, 'ab') as log_file:
            process_options['stdout'] = log_file
            process = subprocess.Popen(command, **process_options)
    except Exception as error:
        model.status = 'failed'
        model.error_message = f'Could not start training worker: {error}'
        model.save(update_fields=['status', 'error_message', 'updated_at'])
        raise

    logger.info(
        'Started model training worker %s for model %s; log: %s',
        process.pid,
        model_id,
        log_path,
    )
    return {'status': 'started', 'model_id': model_id, 'process_id': process.pid}


def run_model_training(model_id):
    """Run training synchronously inside the dedicated background process."""
    model = None
    try:
        from ml_models.evaluate import ModelEvaluator
        from ml_models.train_model import ModelTrainer, TrainingConfig
        from ml_models.cnn_model import AlzheimerCNN
        from ml_models.transfer_learning import TransferLearningModel

        model = TrainedModel.objects.get(pk=model_id)
        model.status = 'training'
        model.error_message = None
        model.save(update_fields=['status', 'error_message', 'updated_at'])

        dataset_split = model.dataset_split
        if not dataset_split and model.dataset:
            dataset_split = model.dataset.splits.filter(is_active=True).first()
        if not dataset_split:
            raise ValueError('No dataset split found.')

        if model.model_type in TransferLearningModel.AVAILABLE_MODELS:
            cnn_model = TransferLearningModel(
                base_model_name=model.model_type,
                num_classes=4,
                class_names=[
                    'Non-Demented',
                    'Very Mild Demented',
                    'Mild Demented',
                    'Moderate Demented',
                ],
            )
            cnn_model.build_model(trainable_base=False)
        else:
            cnn_model = AlzheimerCNN(num_classes=4)
            cnn_model.build_model()

        training_options = model.training_config or {}
        training_config = TrainingConfig(
            batch_size=training_options.get('batch_size', 32),
            epochs=training_options.get('epochs', 50),
            learning_rate=training_options.get('learning_rate', 0.001),
            validation_split=training_options.get('validation_split', 0.2),
            early_stopping_patience=training_options.get('early_stopping_patience', 15),
            reduce_lr_patience=training_options.get('reduce_lr_patience', 5),
            augmentation=training_options.get('use_augmentation', True),
            optimizer=training_options.get('optimizer', 'adam'),
            mixed_precision=training_options.get('mixed_precision', False),
        )
        trainer = ModelTrainer(cnn_model, training_config)
        model.model_config = {
            'type': model.model_type,
            'input_shape': cnn_model.input_shape,
            'num_classes': cnn_model.num_classes,
            'class_names': cnn_model.class_names,
        }
        model.save(update_fields=['model_config', 'updated_at'])

        model_save_path = os.path.join(
            settings.MEDIA_ROOT,
            'models',
            f'model_{model.id}.h5',
        )
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        history = trainer.train(
            train_dir=dataset_split.train_path,
            val_dir=dataset_split.val_path,
            model_save_path=model_save_path,
        )

        cnn_model.save(model_save_path)
        model.file.name = f'models/model_{model.id}.h5'
        model.training_history = history
        model.epochs_trained = len(history.get('accuracy', []))
        model.training_time = trainer.training_time
        model.model_config['class_names'] = cnn_model.class_names

        evaluator = ModelEvaluator(cnn_model)
        metrics = evaluator.evaluate_from_files(dataset_split.test_path)
        model.accuracy = metrics.accuracy
        model.precision = metrics.precision
        model.recall = metrics.recall
        model.f1_score = metrics.f1_score
        model.confusion_matrix = metrics.confusion_matrix
        model.classification_report = metrics.classification_report
        model.per_class_metrics = metrics.per_class_metrics
        model.auc_score = metrics.auc_score
        model.model_summary = cnn_model.summary()
        model.status = 'completed'
        model.error_message = None
        model.save()
        logger.info(
            'Model %s training completed with test accuracy %s',
            model_id,
            model.accuracy,
        )
    except Exception as error:
        logger.exception('Model training failed for model %s', model_id)
        if model is not None:
            TrainedModel.objects.filter(pk=model_id).update(
                status='failed',
                error_message=str(error)[:2000],
            )
        raise


def batch_prediction_task(batch_id):
    """
    Process batch prediction - Simplified version without Celery
    """
    import threading
    
    def process_in_background():
        try:
            batch = BatchPrediction.objects.get(id=batch_id)
            batch.status = 'processing'
            batch.started_at = timezone.now()
            batch.save()
            
            logger.info(f"Processing batch prediction {batch_id}")
            
            # Extract zip file
            extract_path = os.path.join(
                settings.MEDIA_ROOT,
                'temp',
                f'batch_{batch.id}'
            )
            os.makedirs(extract_path, exist_ok=True)
            
            with zipfile.ZipFile(batch.zip_file.path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Get all images
            image_files = []
            for root, dirs, files in os.walk(extract_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_files.append(os.path.join(root, file))
            
            batch.total_images = len(image_files)
            batch.save()
            
            # Load model
            model = batch.model
            
            if model.file and os.path.exists(model.file.path):
                predictor = load_predictor_for_model(model)
            else:
                raise ValueError("Model file not found")
            
            # Process images
            results = []
            stage_counts = {stage[0]: 0 for stage in Prediction.STAGE_CHOICES}
            total_confidence = 0
            
            for i, image_path in enumerate(image_files):
                try:
                    # Predict
                    result = predictor.predict(image_path)
                    if not result.success:
                        raise RuntimeError(result.error_message or 'Model prediction failed.')
                    
                    stage = normalize_prediction_stage(result.predicted_class)
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    total_confidence += result.confidence
                    
                    results.append({
                        'filename': os.path.basename(image_path),
                        'stage': stage,
                        'confidence': result.confidence,
                        'probabilities': result.all_probabilities
                    })
                    
                    batch.processed_images = i + 1
                    batch.save()
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    batch.failed_images += 1
                    results.append({
                        'filename': os.path.basename(image_path),
                        'error': str(e)
                    })
            
            # Calculate statistics
            avg_confidence = total_confidence / batch.processed_images if batch.processed_images > 0 else 0
            
            batch.results_summary = {
                'stage_distribution': stage_counts,
                'average_confidence': avg_confidence,
                'success_rate': (batch.processed_images / batch.total_images * 100) if batch.total_images > 0 else 0,
                'results': results
            }
            
            # Save results file
            results_path = os.path.join(
                settings.MEDIA_ROOT,
                'batch_predictions',
                'results',
                f'batch_{batch.id}_results.json'
            )
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(batch.results_summary, f, indent=2)
            
            with open(results_path, 'rb') as f:
                batch.results_file.save(
                    f'batch_{batch.id}_results.json',
                    ContentFile(f.read()),
                    save=False
                )
            
            batch.status = 'completed'
            batch.completed_at = timezone.now()
            batch.processing_time = (batch.completed_at - batch.started_at).total_seconds()
            batch.save()
            
            # Cleanup
            import shutil
            shutil.rmtree(extract_path, ignore_errors=True)
            
            logger.info(f"Batch prediction {batch_id} completed")
            
            return {
                'batch_id': batch.id,
                'processed': batch.processed_images,
                'failed': batch.failed_images,
                'total': batch.total_images
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            
            try:
                batch = BatchPrediction.objects.get(id=batch_id)
                batch.status = 'failed'
                batch.error_message = str(e)
                batch.save()
            except:
                pass
            
            raise e
    
    # Start processing in background thread
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    return {"status": "started", "batch_id": batch_id}


def update_model_performance_logs():
    """
    Update model performance logs
    """
    from django.utils import timezone
    from datetime import timedelta
    from django.db.models import Avg
    
    today = timezone.localdate()
    yesterday = today - timedelta(days=1)
    
    models = TrainedModel.objects.filter(status='completed')
    count = 0
    
    for model in models:
        # Get yesterday's predictions
        predictions = Prediction.objects.filter(
            model=model,
            created_at__date=yesterday
        )
        
        if predictions.exists():
            total = predictions.count()
            avg_confidence = predictions.aggregate(Avg('confidence'))['confidence__avg']
            avg_time = predictions.aggregate(Avg('processing_time'))['processing_time__avg']
            
            # Stage distribution
            stage_dist = {}
            for stage, _ in Prediction.STAGE_CHOICES:
                count_pred = predictions.filter(predicted_stage=stage).count()
                stage_dist[stage] = count_pred
            
            # Refresh the daily snapshot when this task is retried.
            ModelPerformanceLog.objects.update_or_create(
                model=model,
                date=yesterday,
                defaults={
                    'predictions_count': total,
                    'average_confidence': avg_confidence,
                    'average_processing_time': avg_time,
                    'stage_distribution': stage_dist,
                }
            )
            count += 1
    
    return f"Updated performance logs for {count} models"


def cleanup_old_predictions(days=30):
    """
    Clean up old prediction records and files
    """
    from datetime import timedelta
    
    cutoff_date = timezone.now() - timedelta(days=days)
    
    # Get old predictions
    old_predictions = Prediction.objects.filter(created_at__lt=cutoff_date)
    
    count = 0
    for prediction in old_predictions:
        # Delete image files
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
        count += 1
    
    return f"Cleaned up {count} old predictions"


def export_predictions_csv(user_id, date_from=None, date_to=None):
    """
    Export predictions as CSV
    """
    import csv
    from io import StringIO
    from django.contrib.auth.models import User
    
    user = User.objects.get(id=user_id)
    predictions = Prediction.objects.filter(created_by=user)
    
    if date_from:
        predictions = predictions.filter(created_at__date__gte=date_from)
    if date_to:
        predictions = predictions.filter(created_at__date__lte=date_to)
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Date', 'Stage', 'Confidence', 'Model', 'Processing Time'])
    
    # Write data
    for p in predictions:
        writer.writerow([
            p.id,
            p.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            p.get_predicted_stage_display(),
            f"{p.confidence:.2f}%",
            p.model.name if p.model else 'N/A',
            f"{p.processing_time:.3f}s" if p.processing_time else 'N/A'
        ])
    
    # Save to file
    filename = f"predictions_export_{timezone.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_path = os.path.join(settings.MEDIA_ROOT, 'exports', filename)
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    with open(export_path, 'w') as f:
        f.write(output.getvalue())
    
    return {
        'filename': filename,
        'count': predictions.count(),
        'path': export_path
    }