import io
import tempfile
import zipfile
from pathlib import Path

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image as PILImage
from unittest.mock import patch

from .importer import import_dataset_archive
from .models import Dataset, DatasetSplit


class DatasetArchiveImportTests(TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.media_settings = override_settings(MEDIA_ROOT=self.temporary_directory.name)
        self.media_settings.enable()
        self.addCleanup(self.media_settings.disable)
        self.user = User.objects.create_user(username='dataset-user', password='test-password')

    def make_archive(self, include_traversal=False):
        archive_buffer = io.BytesIO()
        classes = {
            'MildDemented': 'mild_dementia',
            'ModerateDemented': 'moderated_dementia',
            'NonDemented': 'non_demented',
            'VeryMildDemented': 'very_mild_demented',
        }
        with zipfile.ZipFile(archive_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for class_name in classes:
                for split, count in (('train_images', 3), ('test_images', 1)):
                    for index in range(count):
                        image = io.BytesIO()
                        PILImage.new('RGB', (12, 12), color=(index, 0, 0)).save(
                            image,
                            format='PNG',
                        )
                        archive.writestr(
                            f'{split}/{class_name}/sample-{index}.png',
                            image.getvalue(),
                        )
            if include_traversal:
                archive.writestr(
                    'train_images/../outside.png',
                    b'not an image',
                )
        return archive_buffer.getvalue()

    def create_dataset(self, archive_bytes):
        dataset = Dataset.objects.create(
            name='Kaggle MRI dataset',
            uploaded_by=self.user,
            status='processing',
        )
        dataset.zip_file.save(
            'archive.zip',
            SimpleUploadedFile('archive.zip', archive_bytes, content_type='application/zip'),
        )
        return dataset

    def test_import_maps_kaggle_class_names_and_creates_three_splits(self):
        dataset = self.create_dataset(self.make_archive())

        counts = import_dataset_archive(dataset.pk)

        dataset.refresh_from_db()
        split = dataset.splits.get(is_active=True)
        self.assertEqual(dataset.status, 'ready')
        self.assertEqual(dataset.total_images, 16)
        self.assertEqual(split.train_count, 8)
        self.assertEqual(split.val_count, 4)
        self.assertEqual(split.test_count, 4)
        self.assertEqual(
            {category.category: category.count for category in dataset.categories.all()},
            {
                'mild': 4,
                'moderate': 4,
                'non_demented': 4,
                'very_mild': 4,
            },
        )
        for folder in ('train', 'validation', 'test'):
            path_field = 'val_path' if folder == 'validation' else f'{folder}_path'
            self.assertTrue(Path(getattr(split, path_field)).is_dir())
        self.assertEqual(sum(sum(stage_counts.values()) for stage_counts in counts.values()), 16)

    def test_unsafe_archive_path_is_rejected_and_marked_failed(self):
        dataset = self.create_dataset(self.make_archive(include_traversal=True))

        with self.assertRaisesRegex(ValueError, 'unsafe path'):
            import_dataset_archive(dataset.pk)

        dataset.refresh_from_db()
        self.assertEqual(dataset.status, 'failed')
        self.assertIn('unsafe path', dataset.error_message)
        self.assertFalse(
            Path(self.temporary_directory.name, 'datasets', 'processed', str(dataset.pk)).exists()
        )

    @patch('dataset_manager.views.start_dataset_processing')
    def test_upload_saves_archive_and_starts_processing(self, start_processing):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse('dataset_manager:upload'),
            {
                'name': 'Uploaded MRI set',
                'description': 'Kaggle archive',
                'zip_file': SimpleUploadedFile(
                    'dataset.zip',
                    self.make_archive(),
                    content_type='application/zip',
                ),
            },
        )

        dataset = Dataset.objects.get(name='Uploaded MRI set')
        self.assertRedirects(
            response,
            reverse('dataset_manager:detail', args=[dataset.pk]),
        )
        self.assertEqual(dataset.uploaded_by, self.user)
        self.assertEqual(dataset.status, 'processing')
        start_processing.assert_called_once_with(dataset.pk)

    def test_user_can_view_dataset_list_and_training_form_requires_prepared_dataset(self):
        self.client.force_login(self.user)
        dataset = Dataset.objects.create(
            name='Ready dataset',
            uploaded_by=self.user,
            status='ready',
        )
        DatasetSplit.objects.create(
            dataset=dataset,
            train_path=str(Path(self.temporary_directory.name) / 'train'),
            val_path=str(Path(self.temporary_directory.name) / 'validation'),
            test_path=str(Path(self.temporary_directory.name) / 'test'),
        )

        list_response = self.client.get(reverse('dataset_manager:list'))
        self.assertEqual(list_response.status_code, 200)
        self.assertContains(list_response, 'Ready dataset')

        training_response = self.client.get(reverse('prediction:train_model'))
        self.assertEqual(training_response.status_code, 200)
        self.assertContains(training_response, 'Ready dataset')
        self.assertContains(training_response, 'step="0.00001"')

    @patch('prediction.views.create_activity_log')
    @patch('prediction.views.train_model_task')
    def test_training_submission_uses_the_datasets_active_split(
        self,
        start_training,
        create_activity_log,
    ):
        self.client.force_login(self.user)
        dataset = Dataset.objects.create(
            name='Ready dataset',
            uploaded_by=self.user,
            status='ready',
        )
        split = DatasetSplit.objects.create(
            dataset=dataset,
            train_path=str(Path(self.temporary_directory.name) / 'train'),
            val_path=str(Path(self.temporary_directory.name) / 'validation'),
            test_path=str(Path(self.temporary_directory.name) / 'test'),
        )

        response = self.client.post(
            reverse('prediction:train_model'),
            {
                'name': 'New CNN',
                'description': 'Test training run',
                'model_type': 'cnn',
                'dataset': dataset.pk,
                'epochs': 1,
                'batch_size': 8,
                'learning_rate': 0.001,
                'validation_split': 0.2,
            },
        )

        self.assertEqual(response.status_code, 302)
        model = dataset.trained_models.get(name='New CNN')
        self.assertEqual(model.dataset_split, split)
        start_training.assert_called_once_with(model.pk, config=model.training_config)
        create_activity_log.assert_called_once()

    def test_uploaded_dataset_details_are_private_to_owner(self):
        dataset = Dataset.objects.create(
            name='Private dataset',
            uploaded_by=self.user,
            status='ready',
        )
        other_user = User.objects.create_user(username='other-user', password='test-password')
        self.client.force_login(other_user)

        response = self.client.get(
            reverse('dataset_manager:detail', args=[dataset.pk])
        )

        self.assertEqual(response.status_code, 404)
