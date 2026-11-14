/**
 * Prediction JavaScript for Alzheimer's Disease Prediction System
 * Handles prediction-specific functionality, image upload, and result display
 */

// Prediction Class
class PredictionHandler {
    constructor() {
        this.init();
    }
    
    init() {
        this.initImageUpload();
        this.initModelSelector();
        this.initConfidenceSlider();
        this.initBatchUpload();
        this.initQRScanner();
        this.initPredictionHistory();
    }
    
    // Initialize image upload with preview
    initImageUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('id_image');
        const imagePreview = document.getElementById('imagePreview');
        
        if (!uploadArea || !fileInput) return;
        
        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length) {
                this.handleImageFile(files[0], fileInput, imagePreview);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                this.handleImageFile(e.target.files[0], fileInput, imagePreview);
            }
        });
    }
    
    handleImageFile(file, fileInput, previewContainer) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            this.showError('Please upload a valid image file (JPG, JPEG, PNG, BMP, TIFF)');
            return;
        }
        
        // Validate file size (10MB)
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File size exceeds 10MB limit');
            return;
        }
        
        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewContainer) {
                previewContainer.innerHTML = `
                    <div class="image-preview-wrapper">
                        <img src="${e.target.result}" class="image-preview img-fluid rounded" alt="Preview">
                        <button type="button" class="btn btn-danger btn-sm mt-2" id="removeImageBtn">
                            <i class="fas fa-times"></i> Remove
                        </button>
                    </div>
                `;
                
                document.getElementById('removeImageBtn')?.addEventListener('click', () => {
                    previewContainer.innerHTML = '';
                    fileInput.value = '';
                    document.getElementById('submitBtn').disabled = true;
                });
            }
            
            document.getElementById('submitBtn').disabled = false;
        };
        reader.readAsDataURL(file);
        
        // Update file input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
    }
    
    // Initialize model selector
    initModelSelector() {
        const modelSelect = document.getElementById('id_model');
        const modelInfo = document.getElementById('model-info');
        
        if (modelSelect && modelInfo) {
            modelSelect.addEventListener('change', async (e) => {
                const modelId = e.target.value;
                if (modelId) {
                    try {
                        const response = await fetch(`/api/models/${modelId}/`);
                        const data = await response.json();
                        
                        modelInfo.innerHTML = `
                            <div class="alert alert-info mt-2">
                                <strong>${data.name}</strong><br>
                                Accuracy: ${data.accuracy}% | Type: ${data.type}
                            </div>
                        `;
                    } catch (error) {
                        console.error('Error loading model info:', error);
                    }
                } else {
                    modelInfo.innerHTML = '';
                }
            });
        }
    }
    
    // Initialize confidence slider for threshold
    initConfidenceSlider() {
        const slider = document.getElementById('confidence-threshold');
        const valueDisplay = document.getElementById('threshold-value');
        
        if (slider && valueDisplay) {
            slider.addEventListener('input', (e) => {
                const value = e.target.value;
                valueDisplay.textContent = `${value}%`;
            });
        }
    }
    
    // Initialize batch upload
    initBatchUpload() {
        const batchUpload = document.getElementById('batch-upload-form');
        if (batchUpload) {
            batchUpload.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(batchUpload);
                const submitBtn = batchUpload.querySelector('button[type="submit"]');
                const progressBar = document.getElementById('upload-progress');
                
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
                
                if (progressBar) progressBar.style.display = 'block';
                
                try {
                    const response = await fetch('/api/batch-predict/', {
                        method: 'POST',
                        body: formData,
                        headers: {
                            'X-CSRFToken': this.getCsrfToken()
                        }
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        this.showSuccess('Batch upload started successfully!');
                        setTimeout(() => {
                            window.location.href = `/prediction/batch/${data.batch_id}/`;
                        }, 2000);
                    } else {
                        this.showError(data.error || 'Upload failed');
                    }
                } catch (error) {
                    console.error('Error uploading batch:', error);
                    this.showError('Upload failed. Please try again.');
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-upload"></i> Upload Batch';
                    if (progressBar) progressBar.style.display = 'none';
                }
            });
        }
    }
    
    // Initialize QR scanner
    initQRScanner() {
        const startScannerBtn = document.getElementById('startScanner');
        const qrResult = document.getElementById('qrResult');
        
        if (startScannerBtn && typeof Instascan !== 'undefined') {
            let scanner = null;
            
            startScannerBtn.addEventListener('click', async () => {
                try {
                    const cameras = await Instascan.Camera.getCameras();
                    if (cameras.length === 0) {
                        this.showError('No camera found');
                        return;
                    }
                    
                    const video = document.getElementById('preview');
                    scanner = new Instascan.Scanner({ video });
                    
                    scanner.addListener('scan', (content) => {
                        try {
                            const data = JSON.parse(content);
                            this.displayQRResult(data, qrResult);
                            scanner.stop();
                        } catch (e) {
                            this.displayQRResult({ error: 'Invalid QR code data' }, qrResult);
                        }
                    });
                    
                    await scanner.start(cameras[0]);
                    startScannerBtn.textContent = 'Stop Scanner';
                    startScannerBtn.classList.remove('btn-primary');
                    startScannerBtn.classList.add('btn-danger');
                    
                    startScannerBtn.onclick = () => {
                        scanner.stop();
                        startScannerBtn.textContent = 'Start Camera';
                        startScannerBtn.classList.remove('btn-danger');
                        startScannerBtn.classList.add('btn-primary');
                        this.initQRScanner(); // Reinitialize
                    };
                } catch (error) {
                    console.error('Error starting scanner:', error);
                    this.showError('Could not access camera');
                }
            });
        }
    }
    
    displayQRResult(data, container) {
        if (!container) return;
        
        if (data.error) {
            container.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle"></i> ${data.error}
                </div>
            `;
            return;
        }
        
        if (data.id && data.stage) {
            const stageClass = this.getStageClass(data.stage);
            const confidenceColor = data.confidence >= 85 ? 'success' : (data.confidence >= 70 ? 'warning' : 'danger');
            
            container.innerHTML = `
                <div class="card">
                    <div class="card-body text-center">
                        <span class="badge bg-${stageClass} fs-5 p-3 mb-3">${data.stage_display || data.stage}</span>
                        <div class="progress mt-3" style="height: 25px;">
                            <div class="progress-bar bg-${confidenceColor}" style="width: ${data.confidence}%">
                                ${data.confidence}% Confidence
                            </div>
                        </div>
                        <div class="row mt-4">
                            <div class="col-6">
                                <small class="text-muted">Prediction ID</small>
                                <h6>#${data.id}</h6>
                            </div>
                            <div class="col-6">
                                <small class="text-muted">Date</small>
                                <h6>${new Date(data.date).toLocaleDateString()}</h6>
                            </div>
                        </div>
                        <a href="/prediction/result/${data.id}/" class="btn btn-primary mt-3">
                            <i class="fas fa-eye"></i> View Full Details
                        </a>
                    </div>
                </div>
            `;
        } else {
            container.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i>
                    <pre class="mt-2 mb-0">${JSON.stringify(data, null, 2)}</pre>
                </div>
            `;
        }
    }
    
    // Initialize prediction history filters
    initPredictionHistory() {
        const filterForm = document.getElementById('history-filters');
        if (filterForm) {
            filterForm.addEventListener('submit', (e) => {
                e.preventDefault();
                const formData = new FormData(filterForm);
                const params = new URLSearchParams();
                
                for (let [key, value] of formData.entries()) {
                    if (value) params.append(key, value);
                }
                
                window.location.href = `${window.location.pathname}?${params.toString()}`;
            });
        }
        
        // Export functionality
        const exportBtn = document.getElementById('export-history');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                window.location.href = '/prediction/export/?' + window.location.search.substring(1);
            });
        }
    }
    
    // Make prediction via AJAX
    async makePrediction(imageFile, modelId = null) {
        const formData = new FormData();
        formData.append('image', imageFile);
        if (modelId) formData.append('model_id', modelId);
        
        try {
            const response = await fetch('/prediction/api/predict/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': this.getCsrfToken()
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                return data;
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }
    
    // Get prediction status
    async getPredictionStatus(predictionId) {
        try {
            const response = await fetch(`/prediction/api/status/${predictionId}/`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error getting prediction status:', error);
            return null;
        }
    }
    
    // Cancel prediction
    async cancelPrediction(predictionId) {
        try {
            const response = await fetch(`/prediction/api/cancel/${predictionId}/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': this.getCsrfToken()
                }
            });
            const data = await response.json();
            return data.success;
        } catch (error) {
            console.error('Error canceling prediction:', error);
            return false;
        }
    }
    
    // Show result modal
    showResultModal(result) {
        Swal.fire({
            title: 'Prediction Result',
            html: `
                <div class="text-center">
                    <span class="badge bg-${this.getStageClass(result.stage)} fs-4 p-3 mb-3">
                        ${result.stage_display}
                    </span>
                    <div class="progress mt-3" style="height: 30px;">
                        <div class="progress-bar bg-${this.getConfidenceColor(result.confidence)}" 
                             style="width: ${result.confidence}%">
                            ${result.confidence}% Confidence
                        </div>
                    </div>
                    <div class="mt-4">
                        <button class="btn btn-primary" onclick="window.location.href='/prediction/result/${result.id}/'">
                            View Full Details
                        </button>
                    </div>
                </div>
            `,
            icon: 'success',
            confirmButtonText: 'OK'
        });
    }
    
    getStageClass(stage) {
        const classes = {
            'non_demented': 'success',
            'very_mild': 'info',
            'mild': 'warning',
            'moderate': 'danger'
        };
        return classes[stage] || 'secondary';
    }
    
    getConfidenceColor(confidence) {
        if (confidence >= 85) return 'success';
        if (confidence >= 70) return 'warning';
        return 'danger';
    }
    
    showError(message) {
        Swal.fire({
            title: 'Error',
            text: message,
            icon: 'error',
            confirmButtonText: 'OK'
        });
    }
    
    showSuccess(message) {
        Swal.fire({
            title: 'Success',
            text: message,
            icon: 'success',
            timer: 3000,
            showConfirmButton: false
        });
    }
    
    getCsrfToken() {
        const token = document.querySelector('[name=csrfmiddlewaretoken]');
        if (token) return token.value;
        
        const cookie = document.cookie.split('; ').find(row => row.startsWith('csrftoken='));
        return cookie ? cookie.split('=')[1] : '';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (document.querySelector('.prediction-page')) {
        window.predictionHandler = new PredictionHandler();
    }
});