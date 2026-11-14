import logging
import threading

from .importer import import_dataset_archive

logger = logging.getLogger(__name__)


def start_dataset_processing(dataset_id):
    """Start archive processing without blocking the upload request."""
    thread = threading.Thread(
        target=_process_dataset,
        args=(dataset_id,),
        daemon=True,
    )
    thread.start()


def _process_dataset(dataset_id):
    try:
        import_dataset_archive(dataset_id)
    except Exception:
        logger.exception('Dataset import failed for dataset %s', dataset_id)