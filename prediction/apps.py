from django.apps import AppConfig

class PredictionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'prediction'
    verbose_name = 'Prediction Management'
    
    def ready(self):
        """
        Initialize app when ready
        """
        import prediction.signals  # noqa