/**
 * Chart.js Configuration and Utilities
 * Advanced chart configurations for data visualization
 */

// Chart.js Global Configuration
Chart.defaults.font.family = "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = true;

// Color Palette for Alzheimer's Stages
const stageColors = {
    'non_demented': '#28a745',
    'very_mild': '#17a2b8',
    'mild': '#ffc107',
    'moderate': '#dc3545',
    'Non-Demented': '#28a745',
    'Very Mild Demented': '#17a2b8',
    'Mild Demented': '#ffc107',
    'Moderate Demented': '#dc3545'
};

// Confidence Level Colors
const confidenceColors = {
    high: '#28a745',
    medium: '#ffc107',
    low: '#dc3545'
};

// Custom Chart Plugins
const chartPlugins = {
    // Plugin to add data labels
    dataLabels: {
        id: 'dataLabels',
        afterDatasetsDraw(chart, args, options) {
            const { ctx, data, chartArea: { top, bottom, left, right, width, height } } = chart;
            ctx.save();
            
            data.datasets.forEach((dataset, datasetIndex) => {
                const meta = chart.getDatasetMeta(datasetIndex);
                if (!meta.hidden) {
                    meta.data.forEach((element, index) => {
                        const value = dataset.data[index];
                        if (value !== null && value !== undefined) {
                            const position = element.tooltipPosition();
                            ctx.font = 'bold 12px "Inter"';
                            ctx.fillStyle = '#6c757d';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'bottom';
                            ctx.fillText(value, position.x, position.y - 8);
                        }
                    });
                }
            });
            ctx.restore();
        }
    },
    
    // Plugin for gradient backgrounds
    gradientBackground: {
        id: 'gradientBackground',
        beforeDraw(chart) {
            const { ctx, chartArea: { top, bottom, left, right }, config } = chart;
            if (config.options.plugins?.gradientBackground) {
                const gradient = ctx.createLinearGradient(0, top, 0, bottom);
                gradient.addColorStop(0, 'rgba(102, 126, 234, 0.1)');
                gradient.addColorStop(1, 'rgba(118, 75, 162, 0.05)');
                
                ctx.save();
                ctx.fillStyle = gradient;
                ctx.fillRect(left, top, right - left, bottom - top);
                ctx.restore();
            }
        }
    }
};

// Chart Factory Functions
const ChartFactory = {
    // Create Confidence Distribution Chart
    createConfidenceChart: (elementId, data) => {
        const ctx = document.getElementById(elementId).getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    label: 'Confidence Score (%)',
                    data: Object.values(data),
                    backgroundColor: Object.values(data).map(value => 
                        value >= 85 ? confidenceColors.high : 
                        value >= 70 ? confidenceColors.medium : 
                        confidenceColors.low
                    ),
                    borderRadius: 8,
                    barPercentage: 0.7,
                    categoryPercentage: 0.8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.raw.toFixed(1)}% confidence`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Confidence (%)',
                            font: { weight: 'bold' }
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Stage',
                            font: { weight: 'bold' }
                        },
                        ticks: {
                            callback: (value) => {
                                return value.length > 15 ? value.substring(0, 12) + '...' : value;
                            }
                        }
                    }
                }
            }
        });
    },
    
    // Create Stage Distribution Pie Chart
    createStageDistributionChart: (elementId, data) => {
        const ctx = document.getElementById(elementId).getContext('2d');
        const labels = Object.keys(data);
        const values = Object.values(data);
        
        return new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels.map(l => l.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())),
                datasets: [{
                    data: values,
                    backgroundColor: labels.map(label => stageColors[label] || '#6c757d'),
                    borderWidth: 0,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            usePointStyle: true,
                            font: { size: 12 }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.raw / total) * 100).toFixed(1);
                                return `${context.label}: ${context.raw} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    },
    
    // Create Training History Line Chart
    createTrainingHistoryChart: (elementId, data) => {
        const ctx = document.getElementById(elementId).getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.epochs || Array.from({ length: data.accuracy.length }, (_, i) => i + 1),
                datasets: [
                    {
                        label: 'Training Accuracy',
                        data: data.accuracy,
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 3,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Validation Accuracy',
                        data: data.val_accuracy,
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 3,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Training Loss',
                        data: data.loss,
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y1',
                        pointRadius: 3,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Validation Loss',
                        data: data.val_loss,
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y1',
                        pointRadius: 3,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                let label = context.dataset.label || '';
                                let value = context.raw;
                                if (label.includes('Accuracy')) {
                                    value = (value * 100).toFixed(2) + '%';
                                } else {
                                    value = value.toFixed(4);
                                }
                                return `${label}: ${value}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Accuracy (%)',
                            font: { weight: 'bold' }
                        },
                        min: 0,
                        max: 100,
                        ticks: {
                            callback: (value) => value + '%'
                        }
                    },
                    y1: {
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Loss',
                            font: { weight: 'bold' }
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch',
                            font: { weight: 'bold' }
                        }
                    }
                }
            }
        });
    },
    
    // Create Confusion Matrix Heatmap
    createConfusionMatrixChart: (elementId, matrix, classNames) => {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        // Prepare data for heatmap
        const data = [];
        const maxValue = Math.max(...matrix.flat());
        
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                data.push({
                    x: j,
                    y: i,
                    v: matrix[i][j]
                });
            }
        }
        
        return new Chart(ctx, {
            type: 'matrix',
            data: {
                datasets: [{
                    label: 'Confusion Matrix',
                    data: data,
                    backgroundColor: (context) => {
                        const value = context.dataset.data[context.dataIndex].v;
                        const intensity = value / maxValue;
                        return `rgba(0, 123, 255, ${intensity * 0.8 + 0.2})`;
                    },
                    borderWidth: 1,
                    borderColor: '#ffffff',
                    width: ({ chart }) => (chart.chartArea.width / matrix[0].length) - 2,
                    height: ({ chart }) => (chart.chartArea.height / matrix.length) - 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            title: () => '',
                            label: (context) => {
                                const data = context.dataset.data[context.dataIndex];
                                return `${classNames[data.y]} → ${classNames[data.x]}: ${data.v}`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        ticks: {
                            stepSize: 1,
                            callback: (value) => classNames[value],
                            font: { size: 11 }
                        },
                        title: {
                            display: true,
                            text: 'Predicted Label',
                            font: { weight: 'bold' }
                        }
                    },
                    y: {
                        type: 'linear',
                        ticks: {
                            stepSize: 1,
                            callback: (value) => classNames[value],
                            font: { size: 11 }
                        },
                        title: {
                            display: true,
                            text: 'True Label',
                            font: { weight: 'bold' }
                        }
                    }
                }
            }
        });
    },
    
    // Create Category Distribution Bar Chart
    createCategoryBarChart: (elementId, data) => {
        const ctx = document.getElementById(elementId).getContext('2d');
        const labels = Object.keys(data);
        const values = Object.values(data);
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels.map(l => l.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())),
                datasets: [{
                    label: 'Number of Images',
                    data: values,
                    backgroundColor: labels.map(label => stageColors[label] || '#6c757d'),
                    borderRadius: 8,
                    barPercentage: 0.7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.raw} images`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count',
                            font: { weight: 'bold' }
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Stage',
                            font: { weight: 'bold' }
                        }
                    }
                }
            }
        });
    },
    
    // Create Performance Metrics Radar Chart
    createPerformanceRadarChart: (elementId, metrics) => {
        const ctx = document.getElementById(elementId).getContext('2d');
        
        return new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
                datasets: [{
                    label: 'Model Performance',
                    data: [
                        metrics.accuracy * 100,
                        metrics.precision * 100,
                        metrics.recall * 100,
                        metrics.f1_score * 100,
                        metrics.auc_score * 100 || 0
                    ],
                    backgroundColor: 'rgba(0, 123, 255, 0.2)',
                    borderColor: '#007bff',
                    borderWidth: 2,
                    pointBackgroundColor: '#007bff',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#007bff',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            callback: (value) => value + '%'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.label}: ${context.raw.toFixed(1)}%`
                        }
                    }
                }
            }
        });
    }
};

// Export for global use
window.ChartFactory = ChartFactory;
window.stageColors = stageColors;
window.confidenceColors = confidenceColors;