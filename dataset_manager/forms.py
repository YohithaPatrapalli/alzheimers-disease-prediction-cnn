import zipfile

from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator


class DatasetUploadForm(forms.Form):
    name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Dataset name',
        }),
    )
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Optional dataset details',
        }),
    )
    zip_file = forms.FileField(
        validators=[FileExtensionValidator(['zip'])],
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': '.zip',
        }),
    )

    def clean_zip_file(self):
        uploaded_file = self.cleaned_data['zip_file']
        if uploaded_file.size > 1024 * 1024 * 1024:
            raise ValidationError('The ZIP file must be no larger than 1 GB.')

        uploaded_file.seek(0)
        is_zip = zipfile.is_zipfile(uploaded_file)
        uploaded_file.seek(0)
        if not is_zip:
            raise ValidationError('The uploaded file is not a valid ZIP archive.')
        return uploaded_file