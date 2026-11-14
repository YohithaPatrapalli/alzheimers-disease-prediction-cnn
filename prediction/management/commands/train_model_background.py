from django.core.management.base import BaseCommand

from prediction.tasks import run_model_training


class Command(BaseCommand):
    help = 'Run model training in an independent background process.'

    def add_arguments(self, parser):
        parser.add_argument('model_id', type=int)

    def handle(self, *args, **options):
        run_model_training(options['model_id'])
