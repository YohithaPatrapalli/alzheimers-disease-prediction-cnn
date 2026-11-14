from django.db import migrations


STAGE_ALIASES = {
    'nondemented': 'non_demented',
    'non_demented': 'non_demented',
    'verymild': 'very_mild',
    'very_mild': 'very_mild',
    'verymilddemented': 'very_mild',
    'mild': 'mild',
    'milddemented': 'mild',
    'moderate': 'moderate',
    'moderatedemented': 'moderate',
}


def normalize_stage(value):
    normalized = ''.join(character for character in str(value).casefold() if character.isalnum())
    return STAGE_ALIASES.get(normalized, value)


def normalize_existing_predictions(apps, schema_editor):
    Prediction = apps.get_model('prediction', 'Prediction')
    for prediction in Prediction.objects.all().iterator():
        stage = normalize_stage(prediction.predicted_stage)
        scores = {
            normalize_stage(label): confidence
            for label, confidence in (prediction.confidence_scores or {}).items()
        }
        updates = []
        if stage != prediction.predicted_stage:
            prediction.predicted_stage = stage
            updates.append('predicted_stage')
        if scores != prediction.confidence_scores:
            prediction.confidence_scores = scores
            updates.append('confidence_scores')
        if updates:
            prediction.save(update_fields=updates)


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(normalize_existing_predictions, migrations.RunPython.noop),
    ]
