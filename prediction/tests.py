import io
import os
import tempfile
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.utils import timezone
from django.urls import reverse
from PIL import Image

from ml_models.train_model import ModelTrainer, TrainingConfig
from .models import ModelPerformanceLog, Prediction, TrainedModel
from .utils import load_predictor_for_model


class ModelTrainingDataTests(TestCase):
    def test_training_uses_actual_directory_names_as_output_classes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            train_directory = os.path.join(temporary_directory, 'train')
            val_directory = os.path.join(temporary_directory, 'validation')
            class_names = ('mild', 'moderate', 'non_demented', 'very_mild')

            for directory in (train_directory, val_directory):
                for class_name in class_names:
                    class_directory = os.path.join(directory, class_name)
                    os.makedirs(class_directory)
                    image_path = os.path.join(class_directory, 'sample.png')
                    Image.new('RGB', (16, 16), color='white').save(image_path)

            model = SimpleNamespace(
                class_names=[
                    'Non-Demented',
                    'Very Mild Demented',
                    'Mild Demented',
                    'Moderate Demented',
                ],
                num_classes=4,
            )
            trainer = ModelTrainer(
                model,
                TrainingConfig(batch_size=4, augmentation=False),
            )

            train_generator, val_generator = trainer.prepare_data(
                train_directory,
                val_directory,
                target_size=(16, 16),
            )

            self.assertEqual(model.class_names, list(class_names))
            self.assertEqual(train_generator.samples, 4)
            self.assertEqual(val_generator.samples, 4)
            self.assertEqual(
                train_generator.class_indices,
                {'mild': 0, 'moderate': 1, 'non_demented': 2, 'very_mild': 3},
            )


class PredictionAnalyticsPageTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='analyst', password='test-password')
        self.client.force_login(self.user)
        self.model = TrainedModel.objects.create(
            name='Analytics CNN',
            model_type='cnn',
            status='completed',
            accuracy=0.9,
            created_by=self.user,
        )
        Prediction.objects.create(
            image='predictions/input/analytics-test.png',
            model=self.model,
            predicted_stage='mild',
            confidence=88.5,
            confidence_scores={'mild': 88.5},
            created_by=self.user,
        )

    def test_analytics_page_renders_prediction_metrics(self):
        response = self.client.get(reverse('prediction:analytics'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Prediction Analytics')
        self.assertContains(response, 'Total Predictions')
        self.assertContains(response, 'Analytics CNN')
        self.assertContains(response, '88.5%')
        self.assertContains(response, 'Stage Distribution')


class StaticPredictionFlowTests(TestCase):
    def setUp(self):
        self.media_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.media_directory.cleanup)
        self.media_settings = override_settings(MEDIA_ROOT=self.media_directory.name)
        self.media_settings.enable()
        self.addCleanup(self.media_settings.disable)

        self.user = User.objects.create_user(username='predictor', password='test-password')
        self.client.force_login(self.user)

        self.model = TrainedModel.objects.create(
            name='Test CNN',
            model_type='cnn',
            status='completed',
            accuracy=0.9,
            created_by=self.user,
        )
        self.model.file.name = 'models/test-model.h5'
        os.makedirs(os.path.dirname(self.model.file.path), exist_ok=True)
        with open(self.model.file.path, 'wb') as model_file:
            model_file.write(b'test model')
        self.model.save(update_fields=['file'])

    def make_image_upload(self):
        image_buffer = io.BytesIO()
        Image.new('RGB', (120, 120), color='white').save(image_buffer, format='PNG')
        return SimpleUploadedFile('mri.png', image_buffer.getvalue(), content_type='image/png')

    def test_print_report_includes_qr_link_for_owner(self):
        prediction = Prediction.objects.create(
            image='predictions/input/print-report.png',
            model=self.model,
            predicted_stage='mild',
            confidence=87.5,
            confidence_scores={'mild': 87.5, 'non_demented': 12.5},
            created_by=self.user,
        )

        response = self.client.get(
            reverse('prediction:print_report', args=[prediction.pk])
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Print Report')
        self.assertContains(response, 'Mild Demented')
        self.assertContains(response, 'Login-protected report QR code')
        prediction.refresh_from_db()
        self.assertIn(f'/prediction/report/{prediction.pk}/', prediction.qr_data)
        self.assertTrue(prediction.qr_code)
        self.assertTrue(os.path.exists(prediction.qr_code.path))

    def test_print_report_requires_login_and_prediction_ownership(self):
        prediction = Prediction.objects.create(
            image='predictions/input/private-report.png',
            model=self.model,
            predicted_stage='mild',
            confidence=87.5,
            confidence_scores={'mild': 87.5},
            created_by=self.user,
        )
        report_url = reverse('prediction:print_report', args=[prediction.pk])
        self.client.logout()

        anonymous_response = self.client.get(report_url)
        self.assertEqual(anonymous_response.status_code, 302)

        other_user = User.objects.create_user(username='other-viewer', password='test-password')
        self.client.force_login(other_user)
        other_user_response = self.client.get(report_url)
        self.assertEqual(other_user_response.status_code, 404)

    def test_model_details_render_auc_score(self):
        self.model.auc_score = 0.9
        self.model.save(update_fields=['auc_score'])

        response = self.client.get(
            reverse('prediction:model_detail', args=[self.model.pk])
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'AUC Score: 0.900')

    def test_model_details_show_training_failure_reason(self):
        self.model.status = 'failed'
        self.model.error_message = 'Worker interrupted before the first epoch.'
        self.model.save(update_fields=['status', 'error_message'])

        response = self.client.get(
            reverse('prediction:model_detail', args=[self.model.pk])
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Training failed:')
        self.assertContains(response, 'Worker interrupted before the first epoch.')

    @patch('prediction.tasks.subprocess.Popen')
    def test_training_task_launches_a_separate_background_process(self, popen):
        popen.return_value.pid = 12345

        from .tasks import train_model_task

        result = train_model_task(self.model.pk, config={'epochs': 1})

        self.model.refresh_from_db()
        self.assertEqual(result['status'], 'started')
        self.assertEqual(result['process_id'], 12345)
        self.assertEqual(self.model.status, 'pending')
        self.assertEqual(self.model.training_config, {'epochs': 1})
        command = popen.call_args.args[0]
        self.assertEqual(command[-2:], ['train_model_background', str(self.model.pk)])

    @patch('ml_models.predict.Predictor')
    def test_model_without_saved_class_names_uses_training_folder_order(self, predictor_class):
        training_directory = os.path.join(self.media_directory.name, 'training')
        for folder in ('very_mild', 'non_demented', 'moderate', 'mild'):
            os.makedirs(os.path.join(training_directory, folder))

        predictor = predictor_class.return_value
        predictor.model.output_shape = (None, 4)
        model = SimpleNamespace(
            pk=42,
            file=SimpleNamespace(path=os.path.join(self.media_directory.name, 'model.h5')),
            model_config={},
            dataset_split=SimpleNamespace(train_path=training_directory),
        )

        loaded_predictor = load_predictor_for_model(model)

        self.assertIs(loaded_predictor, predictor)
        self.assertEqual(
            loaded_predictor.class_names,
            ['Mild Demented', 'Moderate Demented', 'Non-Demented', 'Very Mild Demented'],
        )

    def test_multiple_predictions_update_the_same_daily_performance_log(self):
        for index, confidence in enumerate((70.0, 90.0)):
            Prediction.objects.create(
                image=f'predictions/input/test-{index}.png',
                model=self.model,
                predicted_stage='mild',
                confidence=confidence,
                confidence_scores={'mild': confidence},
                created_by=self.user,
            )

        logs = ModelPerformanceLog.objects.filter(model=self.model)
        self.assertEqual(logs.count(), 1)
        self.assertEqual(logs.get().predictions_count, 2)
        self.assertEqual(logs.get().average_confidence, 80.0)

    def test_scheduled_performance_update_can_be_retried_for_yesterday(self):
        prediction = Prediction.objects.create(
            image='predictions/input/yesterday.png',
            model=self.model,
            predicted_stage='mild',
            confidence=80.0,
            confidence_scores={'mild': 80.0},
            processing_time=0.5,
            created_by=self.user,
        )
        yesterday = timezone.localdate() - timedelta(days=1)
        Prediction.objects.filter(pk=prediction.pk).update(
            created_at=timezone.now() - timedelta(days=1)
        )

        from .tasks import update_model_performance_logs

        update_model_performance_logs()
        update_model_performance_logs()

        log = ModelPerformanceLog.objects.get(model=self.model, date=yesterday)
        self.assertEqual(log.predictions_count, 1)
        self.assertEqual(log.average_confidence, 80.0)

    @patch('ml_models.predict.Predictor')
    def test_jfif_upload_can_be_predicted(self, predictor_class):
        image_buffer = io.BytesIO()
        Image.new('RGB', (120, 120), color='white').save(image_buffer, format='JPEG')
        predictor_class.return_value.predict.return_value = SimpleNamespace(
            success=True,
            predicted_class='Mild Demented',
            confidence=80.0,
            all_probabilities={'Mild Demented': 80.0},
            processing_time=0.25,
        )

        response = self.client.post(
            reverse('prediction:static_prediction'),
            {
                'image': SimpleUploadedFile(
                    'mri.jfif',
                    image_buffer.getvalue(),
                    content_type='image/jpeg',
                ),
                'model': self.model.pk,
            },
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Mild Demented')
        self.assertEqual(Prediction.objects.get(created_by=self.user).predicted_stage, 'mild')

    def test_invalid_image_dimensions_are_displayed(self):
        image_buffer = io.BytesIO()
        Image.new('RGB', (50, 50), color='white').save(image_buffer, format='PNG')

        response = self.client.post(
            reverse('prediction:static_prediction'),
            {'image': SimpleUploadedFile(
                'small-mri.png',
                image_buffer.getvalue(),
                content_type='image/png',
            )},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Image dimensions too small. Minimum 100x100 pixels.')

    @patch('ml_models.predict.Predictor')
    def test_prediction_output_and_history_are_saved_with_valid_stage_values(self, predictor_class):
        predictor_class.return_value.predict.return_value = SimpleNamespace(
            success=True,
            predicted_class='Very Mild Demented',
            confidence=82.5,
            all_probabilities={
                'Non-Demented': 5.0,
                'Very Mild Demented': 82.5,
                'Mild Demented': 10.0,
                'Moderate Demented': 2.5,
            },
            processing_time=0.25,
        )

        response = self.client.post(
            reverse('prediction:static_prediction'),
            {
                'image': self.make_image_upload(),
                'model': self.model.pk,
                'generate_qr': 'on',
            },
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        prediction = Prediction.objects.get(created_by=self.user)
        self.assertEqual(prediction.predicted_stage, 'very_mild')
        self.assertEqual(prediction.confidence_scores['very_mild'], 82.5)
        self.assertTrue(prediction.qr_code)
        self.assertTrue(os.path.exists(prediction.qr_code.path))
        self.assertContains(response, 'Very Mild Demented')

        history_response = self.client.get(reverse('prediction:history'))
        self.assertEqual(history_response.status_code, 200)
        self.assertContains(history_response, f'#{prediction.pk}')
        self.assertContains(history_response, 'Very Mild Demented')

        detail_response = self.client.get(
            reverse('prediction:history_detail', args=[prediction.pk])
        )
        self.assertEqual(detail_response.status_code, 200)

    @patch('ml_models.predict.Predictor')
    def test_failed_model_inference_does_not_create_empty_history_entry(self, predictor_class):
        predictor_class.return_value.predict.return_value = SimpleNamespace(
            success=False,
            error_message='inference failed',
        )

        response = self.client.post(
            reverse('prediction:static_prediction'),
            {'image': self.make_image_upload(), 'model': self.model.pk},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'inference failed')
        self.assertFalse(Prediction.objects.filter(created_by=self.user).exists())
