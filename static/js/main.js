/**
 * Main JavaScript for Alzheimer's Disease Prediction System
 * Core functionality, utilities, and global event handlers
 */

// Global variables
window.ADSystem = window.ADSystem || {};

// Main Application Class
class AlzheimerSystem {
    constructor() {
        this.init();
    }
    
    init() {
        this.initTooltips();
        this.initPopovers();
        this.initModals();
        this.initFormValidation();
        this.initCopyButtons();
        this.initSearchFilters();
        this.initDarkMode();
        this.initNotifications();
        this.initSidebar();
        this.initKeyboardShortcuts();
    }
    
    // Initialize Bootstrap tooltips
    initTooltips() {
        const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        if (tooltips.length && typeof bootstrap !== 'undefined') {
            tooltips.forEach(tooltip => new bootstrap.Tooltip(tooltip));
        }
    }
    
    // Initialize Bootstrap popovers
    initPopovers() {
        const popovers = document.querySelectorAll('[data-bs-toggle="popover"]');
        if (popovers.length && typeof bootstrap !== 'undefined') {
            popovers.forEach(popover => new bootstrap.Popover(popover));
        }
    }
    
    // Initialize modals
    initModals() {
        const modals = document.querySelectorAll('.modal');
        if (modals.length && typeof bootstrap !== 'undefined') {
            modals.forEach(modal => new bootstrap.Modal(modal));
        }
    }
    
    // Initialize form validation
    initFormValidation() {
        const forms = document.querySelectorAll('.needs-validation');
        forms.forEach(form => {
            form.addEventListener('submit', (event) => {
                if (!form.checkValidity()) {
                    event.preventDefault();
                    event.stopPropagation();
                }
                form.classList.add('was-validated');
            });
        });
    }
    
    // Initialize copy to clipboard buttons
    initCopyButtons() {
        const copyBtns = document.querySelectorAll('.copy-btn');
        copyBtns.forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const text = btn.getAttribute('data-copy') || btn.getAttribute('data-text') || btn.innerText;
                try {
                    await navigator.clipboard.writeText(text);
                    this.showToast('Copied to clipboard!', 'success');
                    this.animateButton(btn);
                } catch (err) {
                    console.error('Failed to copy:', err);
                    this.showToast('Failed to copy', 'error');
                }
            });
        });
    }
    
    // Initialize search filters
    initSearchFilters() {
        const searchInputs = document.querySelectorAll('.search-filter');
        searchInputs.forEach(input => {
            input.addEventListener('input', debounce((e) => {
                const filterValue = e.target.value.toLowerCase();
                const targetSelector = input.getAttribute('data-target');
                const items = document.querySelectorAll(targetSelector);
                
                items.forEach(item => {
                    const text = item.textContent.toLowerCase();
                    if (text.includes(filterValue)) {
                        item.style.display = '';
                    } else {
                        item.style.display = 'none';
                    }
                });
            }, 300));
        });
    }
    
    // Initialize dark mode
    initDarkMode() {
        const darkModeToggle = document.getElementById('darkModeToggle');
        if (darkModeToggle) {
            // Load saved preference
            const isDark = localStorage.getItem('darkMode') === 'true';
            if (isDark) {
                document.body.classList.add('dark-mode');
                this.updateDarkModeIcon(darkModeToggle, true);
            }
            
            darkModeToggle.addEventListener('click', () => {
                const isDarkMode = document.body.classList.toggle('dark-mode');
                localStorage.setItem('darkMode', isDarkMode);
                this.updateDarkModeIcon(darkModeToggle, isDarkMode);
                
                // Dispatch event for other components
                document.dispatchEvent(new CustomEvent('darkModeChanged', { detail: { isDark: isDarkMode } }));
            });
        }
    }
    
    updateDarkModeIcon(toggle, isDark) {
        const icon = toggle.querySelector('i');
        if (isDark) {
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        } else {
            icon.classList.remove('fa-sun');
            icon.classList.add('fa-moon');
        }
    }
    
    // Initialize notifications
    initNotifications() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/notifications/`;
        
        if ('WebSocket' in window && window.userId) {
            try {
                const socket = new WebSocket(wsUrl);
                
                socket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.showNotification(data);
                    this.updateNotificationBadge(data);
                };
                
                socket.onclose = () => {
                    console.log('WebSocket connection closed');
                    // Attempt to reconnect after 5 seconds
                    setTimeout(() => this.initNotifications(), 5000);
                };
                
                window.notificationSocket = socket;
            } catch (e) {
                console.error('WebSocket connection failed:', e);
            }
        }
        
        // Mark all as read button
        const markAllRead = document.getElementById('markAllRead');
        if (markAllRead) {
            markAllRead.addEventListener('click', () => {
                fetch('/api/notifications/mark-all-read/', {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': this.getCsrfToken()
                    }
                }).then(() => {
                    this.updateNotificationBadge({ unread_count: 0 });
                    const badge = document.querySelector('.notification-badge');
                    if (badge) badge.remove();
                });
            });
        }
    }
    
    showNotification(data) {
        // Show toast notification
        this.showToast(data.message, data.type || 'info');
        
        // Update notification dropdown
        this.addNotificationToDropdown(data);
    }
    
    addNotificationToDropdown(data) {
        const container = document.querySelector('.notification-list');
        if (!container) return;
        
        const notificationHtml = `
            <div class="notification-item unread" data-id="${data.id}">
                <div class="d-flex">
                    <div class="flex-shrink-0">
                        <i class="fas ${this.getNotificationIcon(data.type)} ${this.getNotificationColor(data.type)}"></i>
                    </div>
                    <div class="flex-grow-1 ms-3">
                        <p class="mb-1"><strong>${data.title}</strong></p>
                        <p class="mb-1 small">${data.message}</p>
                        <small class="text-muted">just now</small>
                    </div>
                </div>
            </div>
        `;
        
        container.insertAdjacentHTML('afterbegin', notificationHtml);
        
        // Remove old notifications if more than 10
        const items = container.querySelectorAll('.notification-item');
        if (items.length > 10) {
            items[items.length - 1].remove();
        }
    }
    
    updateNotificationBadge(data) {
        const badge = document.querySelector('.notification-badge');
        const unreadCount = data.unread_count || (badge ? parseInt(badge.textContent) + 1 : 1);
        
        if (unreadCount > 0) {
            if (badge) {
                badge.textContent = unreadCount;
                badge.style.display = '';
            } else {
                const notificationLink = document.querySelector('.nav-link[data-bs-toggle="dropdown"][href*="notifications"]');
                if (notificationLink) {
                    const newBadge = document.createElement('span');
                    newBadge.className = 'position-absolute top-0 start-100 translate-middle badge rounded-pill bg-danger notification-badge';
                    newBadge.textContent = unreadCount;
                    notificationLink.style.position = 'relative';
                    notificationLink.appendChild(newBadge);
                }
            }
        } else if (badge) {
            badge.remove();
        }
    }
    
    getNotificationIcon(type) {
        const icons = {
            'success': 'fa-check-circle',
            'error': 'fa-exclamation-circle',
            'warning': 'fa-exclamation-triangle',
            'info': 'fa-info-circle'
        };
        return icons[type] || 'fa-info-circle';
    }
    
    getNotificationColor(type) {
        const colors = {
            'success': 'text-success',
            'error': 'text-danger',
            'warning': 'text-warning',
            'info': 'text-info'
        };
        return colors[type] || 'text-info';
    }
    
    // Initialize sidebar
    initSidebar() {
        const sidebar = document.getElementById('sidebar');
        const toggleBtn = document.getElementById('toggleSidebarBtn');
        
        if (sidebar && toggleBtn) {
            // Load saved state
            const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
            if (isCollapsed) {
                sidebar.classList.add('collapsed');
                this.updateSidebarButton(toggleBtn, true);
            }
            
            toggleBtn.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
                const isNowCollapsed = sidebar.classList.contains('collapsed');
                localStorage.setItem('sidebarCollapsed', isNowCollapsed);
                this.updateSidebarButton(toggleBtn, isNowCollapsed);
                
                // Dispatch event for other components
                document.dispatchEvent(new CustomEvent('sidebarToggled', { detail: { collapsed: isNowCollapsed } }));
            });
        }
        
        // Mobile sidebar toggle
        const mobileToggle = document.querySelector('.navbar-toggler');
        if (mobileToggle && sidebar) {
            mobileToggle.addEventListener('click', () => {
                sidebar.classList.toggle('mobile-open');
            });
        }
    }
    
    updateSidebarButton(button, isCollapsed) {
        const icon = button.querySelector('i');
        const span = button.querySelector('span');
        
        if (isCollapsed) {
            icon.classList.remove('fa-angle-left');
            icon.classList.add('fa-angle-right');
            if (span) span.textContent = ' Expand';
        } else {
            icon.classList.remove('fa-angle-right');
            icon.classList.add('fa-angle-left');
            if (span) span.textContent = ' Collapse Menu';
        }
    }
    
    // Initialize keyboard shortcuts
    initKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl + K: Focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.querySelector('input[type="search"], .search-input');
                if (searchInput) searchInput.focus();
            }
            
            // Esc: Close modals
            if (e.key === 'Escape') {
                const openModal = document.querySelector('.modal.show');
                if (openModal && bootstrap.Modal.getInstance(openModal)) {
                    bootstrap.Modal.getInstance(openModal).hide();
                }
            }
            
            // Ctrl + /: Show shortcuts help
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                this.showShortcutsHelp();
            }
        });
    }
    
    showShortcutsHelp() {
        Swal.fire({
            title: 'Keyboard Shortcuts',
            html: `
                <div class="text-start">
                    <p><kbd>Ctrl</kbd> + <kbd>K</kbd> - Focus search</p>
                    <p><kbd>Esc</kbd> - Close modal</p>
                    <p><kbd>Ctrl</kbd> + <kbd>/</kbd> - Show this help</p>
                    <p><kbd>Ctrl</kbd> + <kbd>P</kbd> - New prediction</p>
                </div>
            `,
            icon: 'info',
            confirmButtonText: 'Got it'
        });
    }
    
    // Show toast notification
    showToast(message, type = 'info') {
        const toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            // Create container if not exists
            const container = document.createElement('div');
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            document.body.appendChild(container);
        }
        
        const toastId = 'toast-' + Date.now();
        const toastHtml = `
            <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true" data-bs-autohide="true" data-bs-delay="5000">
                <div class="toast-header bg-${type} text-white">
                    <i class="fas ${this.getNotificationIcon(type)} me-2"></i>
                    <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        const container = document.querySelector('.toast-container');
        container.insertAdjacentHTML('beforeend', toastHtml);
        
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement);
        toast.show();
        
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }
    
    // Animate button click
    animateButton(button) {
        button.classList.add('btn-pulse');
        setTimeout(() => button.classList.remove('btn-pulse'), 300);
    }
    
    // Get CSRF token
    getCsrfToken() {
        const token = document.querySelector('[name=csrfmiddlewaretoken]');
        if (token) return token.value;
        
        const cookie = document.cookie.split('; ').find(row => row.startsWith('csrftoken='));
        return cookie ? cookie.split('=')[1] : '';
    }
}

// Utility: Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Utility: Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Utility: Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Utility: Format date
function formatDate(date, format = 'short') {
    const d = new Date(date);
    if (format === 'short') {
        return d.toLocaleDateString();
    } else if (format === 'long') {
        return d.toLocaleString();
    } else if (format === 'relative') {
        const now = new Date();
        const diff = Math.floor((now - d) / 1000);
        
        if (diff < 60) return 'just now';
        if (diff < 3600) return `${Math.floor(diff / 60)} minutes ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)} hours ago`;
        if (diff < 604800) return `${Math.floor(diff / 86400)} days ago`;
        return d.toLocaleDateString();
    }
    return d.toString();
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.ADSystem = new AlzheimerSystem();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.notificationSocket) {
        window.notificationSocket.close();
    }
    if (window.dashboard) {
        window.dashboard.stopAutoRefresh();
    }
});