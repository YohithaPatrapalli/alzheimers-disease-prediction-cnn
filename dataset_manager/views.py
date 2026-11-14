from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Q
from django.shortcuts import redirect, render
from django.views import View
from django.views.generic import DetailView, ListView

from .forms import DatasetUploadForm
from .models import Dataset
from .tasks import start_dataset_processing


class DatasetListView(LoginRequiredMixin, ListView):
    model = Dataset
    template_name = 'dataset_manager/managed_list.html'
    context_object_name = 'datasets'
    paginate_by = 12

    def get_queryset(self):
        queryset = Dataset.objects.filter(uploaded_by=self.request.user).prefetch_related(
            'categories',
            'splits',
        )
        query = self.request.GET.get('search', '').strip()
        status = self.request.GET.get('status', '').strip()
        if query:
            queryset = queryset.filter(
                Q(name__icontains=query) | Q(description__icontains=query)
            )
        if status in dict(Dataset.STATUS_CHOICES):
            queryset = queryset.filter(status=status)
        return queryset


class DatasetUploadView(LoginRequiredMixin, View):
    template_name = 'dataset_manager/import_upload.html'

    def get(self, request):
        return render(request, self.template_name, {'form': DatasetUploadForm()})

    def post(self, request):
        form = DatasetUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return render(request, self.template_name, {'form': form})

        dataset = Dataset.objects.create(
            name=form.cleaned_data['name'],
            description=form.cleaned_data['description'],
            zip_file=form.cleaned_data['zip_file'],
            uploaded_by=request.user,
            status='processing',
        )
        try:
            start_dataset_processing(dataset.pk)
        except Exception as error:
            dataset.status = 'failed'
            dataset.error_message = str(error)
            dataset.save(update_fields=['status', 'error_message', 'updated_at'])
            messages.error(request, f'Dataset processing could not start: {error}')
            return redirect('dataset_manager:detail', pk=dataset.pk)

        messages.success(
            request,
            'Dataset uploaded. It will be available for training after processing finishes.',
        )
        return redirect('dataset_manager:detail', pk=dataset.pk)


class DatasetDetailView(LoginRequiredMixin, DetailView):
    model = Dataset
    template_name = 'dataset_manager/import_detail.html'
    context_object_name = 'dataset'

    def get_queryset(self):
        return Dataset.objects.filter(uploaded_by=self.request.user).prefetch_related(
            'categories',
            'splits',
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['active_split'] = self.object.splits.filter(is_active=True).first()
        context['stage_distribution'] = self.object.get_category_distribution()
        context['test_class_counts'] = self._get_test_class_counts(context['active_split'])
        return context

    @staticmethod
    def _get_test_class_counts(dataset_split):
        counts = {}
        if not dataset_split or not dataset_split.test_path:
            return counts
        from pathlib import Path

        for folder in Path(dataset_split.test_path).iterdir():
            if folder.is_dir():
                counts[folder.name] = sum(1 for path in folder.iterdir() if path.is_file())
        return counts