from django import forms
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError
from .models import TrainedModel, Prediction, BatchPrediction, PredictionFeedback
from dataset_manager.models import Dataset
import json


# Custom widget for multiple file upload
class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True
    
    def __init__(self, attrs=None):
        default_attrs = {'multiple': 'multiple'}
        if attrs:
            default_attrs.update(attrs)
        super().__init__(default_attrs)
    
    def value_from_datadict(self, data, files, name):
        if hasattr(files, 'getlist'):
            return files.getlist(name)
        return [files.get(name)]


class MultipleFileField(forms.FileField):
    widget = MultipleFileInput
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('widget', MultipleFileInput())
        super().__init__(*args, **kwargs)
    
    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = []
            for d in data:
                result.append(single_file_clean(d, initial))
            return result
        return single_file_clean(data, initial)


class ModelTrainingForm(forms.Form):
    """Form for training new model"""
    
    MODEL_CHOICES = [
        ('cnn', 'Custom CNN'),
        ('vgg16', 'VGG16'),
        ('resnet50', 'ResNet50'),
        ('inceptionv3', 'InceptionV3'),
        ('densenet121', 'DenseNet121'),
        ('mobilenetv2', 'MobileNetV2'),
        ('efficientnetb0', 'EfficientNetB0')
    ]
    
    name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter model name'
        })
    )
    
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Enter model description',
            'rows': 3
        })
    )
    
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    dataset = forms.ModelChoiceField(
        queryset=Dataset.objects.none(),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    epochs = forms.IntegerField(
        initial=50,
        min_value=1,
        max_value=200,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '1'
        })
    )
    
    batch_size = forms.IntegerField(
        initial=32,
        min_value=8,
        max_value=128,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '8'
        })
    )
    
    learning_rate = forms.FloatField(
        initial=0.001,
        min_value=0.00001,
        max_value=0.1,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.00001'
        })
    )
    
    validation_split = forms.FloatField(
        initial=0.2,
        min_value=0.1,
        max_value=0.4,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.05'
        })
    )
    
    use_augmentation = forms.BooleanField(
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    early_stopping = forms.BooleanField(
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        datasets = Dataset.objects.filter(status='ready')
        if user is not None:
            datasets = datasets.filter(uploaded_by=user)
        self.fields['dataset'].queryset = datasets

    def clean_dataset(self):
        dataset = self.cleaned_data['dataset']
        if not dataset.splits.filter(
            is_active=True,
            train_path__gt='',
            val_path__gt='',
            test_path__gt='',
        ).exists():
            raise ValidationError(
                'This dataset has no complete active train/validation/test split.'
            )
        return dataset


class StaticPredictionForm(forms.Form):
    """Form for single image prediction"""
    
    image = forms.ImageField(
        validators=[FileExtensionValidator(['jpg', 'jpeg', 'jfif', 'png', 'bmp'])],
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*',
            'id': 'image-input'
        })
    )
    
    model = forms.ModelChoiceField(
        queryset=TrainedModel.objects.filter(status='completed'),
        empty_label="Select a model",
        widget=forms.Select(attrs={'class': 'form-select'}),
        required=False,
        help_text="Leave empty to use the latest model"
    )
    
    generate_qr = forms.BooleanField(
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # Check file size (max 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError('Image size should not exceed 10MB.')
            
            # Check dimensions
            from PIL import Image, UnidentifiedImageError
            try:
                img = Image.open(image)
            except (UnidentifiedImageError, OSError) as e:
                raise forms.ValidationError(f'Invalid image file: {str(e)}')

            if img.width < 100 or img.height < 100:
                raise forms.ValidationError('Image dimensions too small. Minimum 100x100 pixels.')
            if img.width > 5000 or img.height > 5000:
                raise forms.ValidationError('Image dimensions too large. Maximum 5000x5000 pixels.')
        
        return image


class DynamicPredictionForm(forms.Form):
    """Form for dynamic prediction with QR code"""
    
    qr_image = forms.ImageField(
        validators=[FileExtensionValidator(['jpg', 'jpeg', 'png'])],
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*',
            'id': 'qr-input'
        })
    )
    
    def clean_qr_image(self):
        qr_image = self.cleaned_data.get('qr_image')
        if qr_image:
            # Validate QR code
            from PIL import Image
            import pyzbar.pyzbar as pyzbar
            
            try:
                img = Image.open(qr_image)
                decoded = pyzbar.decode(img)
                if not decoded:
                    raise forms.ValidationError('No QR code found in the image.')
            except Exception as e:
                raise forms.ValidationError(f'Invalid QR code image: {str(e)}')
        
        return qr_image


class BatchPredictionForm(forms.ModelForm):
    """Form for batch prediction"""
    
    class Meta:
        model = BatchPrediction
        fields = ['name', 'model', 'zip_file']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter batch name'
            }),
            'model': forms.Select(attrs={'class': 'form-select'}),
            'zip_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.zip'
            })
        }
    
    def clean_zip_file(self):
        zip_file = self.cleaned_data.get('zip_file')
        if zip_file:
            # Check file size (max 500MB)
            if zip_file.size > 500 * 1024 * 1024:
                raise forms.ValidationError('ZIP file size should not exceed 500MB.')
        
        return zip_file


class PredictionFeedbackForm(forms.ModelForm):
    """Form for prediction feedback"""
    
    class Meta:
        model = PredictionFeedback
        fields = ['rating', 'correct_stage', 'comments']
        widgets = {
            'rating': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1',
                'max': '5',
                'step': '1'
            }),
            'correct_stage': forms.Select(attrs={'class': 'form-select'}),
            'comments': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your comments or suggestions...',
                'rows': 3
            })
        }


class ModelComparisonForm(forms.Form):
    """Form for comparing multiple models"""
    
    models = forms.ModelMultipleChoiceField(
        queryset=TrainedModel.objects.filter(status='completed'),
        widget=forms.SelectMultiple(attrs={
            'class': 'form-select',
            'size': '5'
        })
    )
    
    # Use custom MultipleFileField for multiple image uploads
    test_images = MultipleFileField(
        required=False,
        validators=[FileExtensionValidator(['jpg', 'jpeg', 'png'])],
        widget=MultipleFileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*'
        })
    )
    
    comparison_type = forms.ChoiceField(
        choices=[
            ('accuracy', 'Compare by Accuracy'),
            ('speed', 'Compare by Speed'),
            ('both', 'Compare Both')
        ],
        initial='both',
        widget=forms.Select(attrs={'class': 'form-select'})
    )


class PredictionHistoryFilterForm(forms.Form):
    """Form for filtering prediction history"""
    
    STAGE_CHOICES = [
        ('', 'All Stages'),
        ('non_demented', 'Non-Demented'),
        ('very_mild', 'Very Mild Demented'),
        ('mild', 'Mild Demented'),
        ('moderate', 'Moderate Demented'),
    ]
    
    date_from = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )
    
    date_to = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )
    
    stage = forms.ChoiceField(
        required=False,
        choices=STAGE_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    min_confidence = forms.IntegerField(
        required=False,
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Min confidence %'
        })
    )
    
    model = forms.ModelChoiceField(
        required=False,
        queryset=TrainedModel.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search by ID...'
        })
    )