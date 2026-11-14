from django.apps import AppConfig

class DatasetManagerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dataset_manager'
    verbose_name = 'Dataset Management'
    
    def ready(self):
        """
        Initialize app when ready
        """
        import dataset_manager.signals  # noqa