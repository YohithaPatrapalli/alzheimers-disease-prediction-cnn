/**
 * Dashboard JavaScript for Alzheimer's Disease Prediction System
 * Handles dashboard-specific functionality, real-time updates, and statistics
 */

// Dashboard Class
class Dashboard {
    constructor() {
        this.refreshInterval = null;
        this.charts = {};
        this.init();
    }
    
    init() {
        this.loadStatistics();
        this.loadRecentActivities();
        this.loadRecentPredictions();
        this.initCharts();
        this.initEventListeners();
        this.startAutoRefresh();
    }
    
    loadStatistics() {
        fetch('/api/dashboard/statistics/')
            .then(response => response.json())
            .then(data => {
                this.updateStatisticsCards(data);
            })
            .catch(error => console.error('Error loading statistics:', error));
    }
    
    updateStatisticsCards(data) {
        const elements = {
            totalPredictions: document.getElementById('total-predictions'),
            totalDatasets: document.getElementById('total-datasets'),
            totalModels: document.getElementById('total-models'),
            avgConfidence: document.getElementById('avg-confidence')
        };
        
        if (elements.totalPredictions) {
            this.animateNumber(elements.totalPredictions, elements.totalPredictions.textContent, data.total_predictions);
        }
        if (elements.totalDatasets) {
            this.animateNumber(elements.totalDatasets, elements.totalDatasets.textContent, data.total_datasets);
        }
        if (elements.totalModels) {
            this.animateNumber(elements.totalModels, elements.totalModels.textContent, data.total_models);
        }
        if (elements.avgConfidence) {
            this.animateNumber(elements.avgConfidence, elements.avgConfidence.textContent, data.avg_confidence);
        }
        
        // Update storage usage
        const storageBar = document.getElementById('storage-usage');
        if (storageBar && data.storage_percentage) {
            storageBar.style.width = `${data.storage_percentage}%`;
            storageBar.textContent = `${data.storage_percentage}%`;
        }
    }
    
    animateNumber(element, start, end, duration = 1000) {
        let startTime = null;
        const startNum = parseInt(start) || 0;
        const endNum = parseInt(end) || 0;
        
        if (startNum === endNum) return;
        
        const animate = (currentTime) => {
            if (!startTime) startTime = currentTime;
            const progress = Math.min((currentTime - startTime) / duration, 1);
            const current = Math.floor(startNum + (endNum - startNum) * progress);
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                element.textContent = endNum;
            }
        };
        
        requestAnimationFrame(animate);
    }
    
    loadRecentActivities() {
        fetch('/api/dashboard/recent-activities/')
            .then(response => response.json())
            .then(data => {
                this.renderActivities(data.activities);
            })
            .catch(error => console.error('Error loading activities:', error));
    }
    
    renderActivities(activities) {
        const container = document.getElementById('recent-activities');
        if (!container) return;
        
        if (!activities || activities.length === 0) {
            container.innerHTML = '<div class="text-center py-4"><i class="fas fa-inbox fa-2x text-muted"></i><p class="mt-2 text-muted">No recent activities</p></div>';
            return;
        }
        
        let html = '<div class="timeline">';
        activities.forEach(activity => {
            const icon = this.getActivityIcon(activity.action);
            const color = this.getActivityColor(activity.action);
            
            html += `
                <div class="timeline-item">
                    <div class="timeline-icon bg-${color}">
                        <i class="fas fa-${icon}"></i>
                    </div>
                    <div class="timeline-content">
                        <div class="d-flex justify-content-between">
                            <strong>${activity.action_display}</strong>
                            <small class="text-muted">${this.formatTimeAgo(activity.timestamp)}</small>
                        </div>
                        <p class="mb-0 text-muted small">${activity.description}</p>
                    </div>
                </div>
            `;
        });
        html += '</div>';
        
        container.innerHTML = html;
    }
    
    getActivityIcon(action) {
        const icons = {
            'login': 'sign-in-alt',
            'upload': 'upload',
            'prediction': 'brain',
            'training': 'robot',
            'delete': 'trash-alt',
            'update': 'edit',
            'export': 'file-export'
        };
        return icons[action] || 'info-circle';
    }
    
    getActivityColor(action) {
        const colors = {
            'login': 'success',
            'upload': 'primary',
            'prediction': 'info',
            'training': 'warning',
            'delete': 'danger',
            'update': 'secondary',
            'export': 'dark'
        };
        return colors[action] || 'secondary';
    }
    
    loadRecentPredictions() {
        fetch('/api/dashboard/recent-predictions/')
            .then(response => response.json())
            .then(data => {
                this.renderPredictions(data.predictions);
            })
            .catch(error => console.error('Error loading predictions:', error));
    }
    
    renderPredictions(predictions) {
        const container = document.getElementById('recent-predictions');
        if (!container) return;
        
        if (!predictions || predictions.length === 0) {
            container.innerHTML = '<div class="text-center py-4"><i class="fas fa-chart-line fa-2x text-muted"></i><p class="mt-2 text-muted">No predictions yet</p><a href="/prediction/static/" class="btn btn-sm btn-primary mt-2">Make a Prediction</a></div>';
            return;
        }
        
        let html = '<div class="list-group list-group-flush">';
        predictions.forEach(pred => {
            const stageClass = this.getStageClass(pred.stage);
            const confidenceColor = pred.confidence >= 85 ? 'success' : (pred.confidence >= 70 ? 'warning' : 'danger');
            
            html += `
                <div class="list-group-item">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <span class="badge bg-${stageClass} me-2">${pred.stage_display}</span>
                            <small class="text-muted">${this.formatTimeAgo(pred.created_at)}</small>
                        </div>
                        <div>
                            <span class="badge bg-${confidenceColor}">${pred.confidence}%</span>
                            <a href="/prediction/result/${pred.id}/" class="btn btn-sm btn-outline-primary ms-2">
                                <i class="fas fa-eye"></i>
                            </a>
                        </div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
        
        container.innerHTML = html;
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
    
    initCharts() {
        // Load prediction trend chart
        this.loadPredictionTrendChart();
        
        // Load stage distribution chart
        this.loadStageDistributionChart();
        
        // Load confidence distribution chart
        this.loadConfidenceDistributionChart();
    }
    
    loadPredictionTrendChart() {
        const canvas = document.getElementById('prediction-trend-chart');
        if (!canvas) return;
        
        fetch('/api/dashboard/prediction-trend/')
            .then(response => response.json())
            .then(data => {
                if (window.ChartFactory) {
                    this.charts.trend = new Chart(canvas.getContext('2d'), {
                        type: 'line',
                        data: {
                            labels: data.labels,
                            datasets: [{
                                label: 'Predictions',
                                data: data.values,
                                borderColor: '#007bff',
                                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                                fill: true,
                                tension: 0.4,
                                pointRadius: 4,
                                pointHoverRadius: 6
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: {
                                legend: { display: false },
                                tooltip: {
                                    callbacks: {
                                        label: (context) => `${context.raw} predictions`
                                    }
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: 'Number of Predictions'
                                    }
                                }
                            }
                        }
                    });
                }
            })
            .catch(error => console.error('Error loading prediction trend:', error));
    }
    
    loadStageDistributionChart() {
        const canvas = document.getElementById('stage-distribution-chart');
        if (!canvas) return;
        
        fetch('/api/dashboard/stage-distribution/')
            .then(response => response.json())
            .then(data => {
                if (window.ChartFactory) {
                    this.charts.stage = window.ChartFactory.createStageDistributionChart('stage-distribution-chart', data);
                }
            })
            .catch(error => console.error('Error loading stage distribution:', error));
    }
    
    loadConfidenceDistributionChart() {
        const canvas = document.getElementById('confidence-distribution-chart');
        if (!canvas) return;
        
        fetch('/api/dashboard/confidence-distribution/')
            .then(response => response.json())
            .then(data => {
                new Chart(canvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: ['High (≥85%)', 'Medium (70-84%)', 'Low (<70%)'],
                        datasets: [{
                            label: 'Predictions',
                            data: [data.high, data.medium, data.low],
                            backgroundColor: ['#28a745', '#ffc107', '#dc3545'],
                            borderRadius: 8
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Number of Predictions'
                                }
                            }
                        }
                    }
                });
            })
            .catch(error => console.error('Error loading confidence distribution:', error));
    }
    
    initEventListeners() {
        // Refresh button
        const refreshBtn = document.getElementById('refresh-dashboard');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadStatistics();
                this.loadRecentActivities();
                this.loadRecentPredictions();
                this.loadPredictionTrendChart();
                this.loadStageDistributionChart();
                this.loadConfidenceDistributionChart();
                
                // Show refresh animation
                refreshBtn.classList.add('fa-spin');
                setTimeout(() => refreshBtn.classList.remove('fa-spin'), 1000);
            });
        }
        
        // Date range picker
        const dateRange = document.getElementById('date-range');
        if (dateRange) {
            dateRange.addEventListener('change', () => {
                this.loadPredictionTrendChart();
            });
        }
    }
    
    startAutoRefresh() {
        // Refresh every 30 seconds
        this.refreshInterval = setInterval(() => {
            this.loadStatistics();
            this.loadRecentActivities();
            this.loadRecentPredictions();
        }, 30000);
    }
    
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
    
    formatTimeAgo(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const seconds = Math.floor((now - date) / 1000);
        
        if (seconds < 60) return 'just now';
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        const days = Math.floor(hours / 24);
        if (days < 7) return `${days} day${days > 1 ? 's' : ''} ago`;
        const weeks = Math.floor(days / 7);
        return `${weeks} week${weeks > 1 ? 's' : ''} ago`;
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (document.querySelector('.dashboard-page')) {
        window.dashboard = new Dashboard();
    }
});