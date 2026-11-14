import os
import random
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from prediction.utils import normalize_prediction_stage

from .models import Dataset, DatasetSplit, ImageCategory


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jfif', '.jpg', '.png', '.tif', '.tiff'}
SOURCE_SPLITS = {'train_images': 'train', 'test_images': 'test'}
EXPECTED_STAGES = {'non_demented', 'very_mild', 'mild', 'moderate'}
MAX_ARCHIVE_UNCOMPRESSED_SIZE = 5 * 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 100000
VALIDATION_FRACTION = 0.2


def _archive_images(archive):
    image_entries = {'train': Counter(), 'test': Counter()}
    entries_by_split_and_stage = {
        split: {stage: [] for stage in EXPECTED_STAGES}
        for split in SOURCE_SPLITS.values()
    }

    if len(archive.infolist()) > MAX_ARCHIVE_ENTRIES:
        raise ValueError(f'The archive contains more than {MAX_ARCHIVE_ENTRIES} entries.')
    if sum(info.file_size for info in archive.infolist()) > MAX_ARCHIVE_UNCOMPRESSED_SIZE:
        raise ValueError('The uncompressed archive exceeds the 5 GB limit.')

    for info in archive.infolist():
        name = info.filename.replace('\\', '/')
        path = PurePosixPath(name)
        if not path.parts:
            continue
        if path.is_absolute() or '\x00' in name or ':' in path.parts[0]:
            raise ValueError('The archive contains an unsafe path.')
        if any(part in {'.', '..'} for part in path.parts):
            raise ValueError('The archive contains an unsafe path.')
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('The archive contains a symbolic link.')
        if info.is_dir() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        if not info.file_size:
            raise ValueError(f'The archive contains an empty image file: {path.name}.')
        if info.flag_bits & 0x1:
            raise ValueError('Password-protected ZIP files are not supported.')

        parts = [part.casefold() for part in path.parts]
        split_index = next(
            (index for index, part in enumerate(parts) if part in SOURCE_SPLITS),
            None,
        )
        if split_index is None or split_index + 2 >= len(parts):
            continue

        source_split = parts[split_index]
        output_split = SOURCE_SPLITS[source_split]
        source_stage = parts[split_index + 1]
        try:
            stage = normalize_prediction_stage(source_stage)
        except ValueError as error:
            raise ValueError(
                f'Unsupported class folder "{path.parts[split_index + 1]}". '
                'Expected Non-Demented, Very Mild, Mild, and Moderate classes.'
            ) from error

        entries_by_split_and_stage[output_split][stage].append(info)
        image_entries[output_split][stage] += 1

    for split in ('train', 'test'):
        missing_stages = EXPECTED_STAGES - set(
            stage for stage, count in image_entries[split].items() if count
        )
        if missing_stages:
            raise ValueError(
                f'The {split} portion of the archive is missing image classes: '
                f'{", ".join(sorted(missing_stages))}.'
            )

    for stage, entries in entries_by_split_and_stage['train'].items():
        if len(entries) < 2:
            raise ValueError(
                f'The training class "{stage}" needs at least two images '
                'to create a validation split.'
            )

    return entries_by_split_and_stage


def _copy_archive_entry(archive, info, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, destination.open('wb') as target:
        shutil.copyfileobj(source, target)


def import_dataset_archive(dataset_id):
    """Extract a four-class Kaggle archive into training, validation and test folders."""
    dataset = Dataset.objects.get(pk=dataset_id)
    destination_root = Path(settings.MEDIA_ROOT) / 'datasets' / 'processed' / str(dataset.pk)
    started_at = timezone.now()
    rng = random.Random(42)

    try:
        if not dataset.zip_file or not os.path.isfile(dataset.zip_file.path):
            raise ValueError('The uploaded dataset archive could not be found.')

        shutil.rmtree(destination_root, ignore_errors=True)
        with zipfile.ZipFile(dataset.zip_file.path) as archive:
            entries = _archive_images(archive)
            split_counts = {
                'train': Counter(),
                'validation': Counter(),
                'test': Counter(),
            }

            for stage in sorted(EXPECTED_STAGES):
                training_entries = list(entries['train'][stage])
                rng.shuffle(training_entries)
                validation_count = max(
                    1,
                    round(len(training_entries) * VALIDATION_FRACTION),
                )
                validation_count = min(validation_count, len(training_entries) - 1)

                assignments = (
                    ('validation', training_entries[:validation_count]),
                    ('train', training_entries[validation_count:]),
                    ('test', entries['test'][stage]),
                )

                for split, assigned_entries in assignments:
                    for index, info in enumerate(assigned_entries):
                        filename = PurePosixPath(
                            info.filename.replace('\\', '/')
                        ).name
                        safe_filename = f'{index:06d}_{filename}'
                        target = destination_root / split / stage / safe_filename
                        _copy_archive_entry(archive, info, target)
                    split_counts[split][stage] = len(assigned_entries)

        total_images = sum(sum(counts.values()) for counts in split_counts.values())
        total_split_count = sum(
            sum(split_counts[split].values()) for split in ('train', 'validation', 'test')
        )
        if total_images != total_split_count or not total_images:
            raise ValueError('The archive did not contain a usable set of labeled images.')

        train_count = sum(split_counts['train'].values())
        validation_count = sum(split_counts['validation'].values())
        test_count = sum(split_counts['test'].values())

        with transaction.atomic():
            dataset.refresh_from_db()
            dataset.categories.all().delete()
            dataset.splits.all().delete()

            for stage in sorted(EXPECTED_STAGES):
                ImageCategory.objects.create(
                    dataset=dataset,
                    category=stage,
                    count=sum(split_counts[split][stage] for split in split_counts),
                )

            DatasetSplit.objects.create(
                dataset=dataset,
                train_ratio=train_count / total_images,
                val_ratio=validation_count / total_images,
                test_ratio=test_count / total_images,
                train_path=str(destination_root / 'train'),
                val_path=str(destination_root / 'validation'),
                test_path=str(destination_root / 'test'),
                train_count=train_count,
                val_count=validation_count,
                test_count=test_count,
                is_active=True,
            )

            dataset.total_images = total_images
            dataset.status = 'ready'
            dataset.error_message = ''
            dataset.processing_time = (timezone.now() - started_at).total_seconds()
            dataset.save(update_fields=[
                'total_images',
                'status',
                'error_message',
                'processing_time',
                'updated_at',
            ])

        return split_counts
    except Exception as error:
        shutil.rmtree(destination_root, ignore_errors=True)
        Dataset.objects.filter(pk=dataset_id).update(
            status='failed',
            error_message=str(error)[:2000],
            processing_time=(timezone.now() - started_at).total_seconds(),
        )
        raise
