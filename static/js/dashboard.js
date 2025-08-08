// Dashboard JavaScript functionality

class Dashboard {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.startPolling();
    }

    setupEventListeners() {
        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('autoRefresh');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startPolling();
                } else {
                    this.stopPolling();
                }
            });
        }

        // Manual refresh button
        const refreshBtn = document.getElementById('refreshBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.refreshData();
            });
        }
    }

    async refreshData() {
        try {
            // Update learning statistics
            const response = await fetch('/api/learning_stats');
            if (response.ok) {
                const stats = await response.json();
                this.updateLearningStats(stats);
            }
        } catch (error) {
            console.error('Error refreshing data:', error);
        }
    }

    updateLearningStats(stats) {
        // Update strategy success rates
        Object.keys(stats.strategies).forEach(strategy => {
            const strategyElement = document.querySelector(`[data-strategy="${strategy}"]`);
            if (strategyElement) {
                const successRate = stats.strategies[strategy].success_rate * 100;
                strategyElement.querySelector('.progress-bar').style.width = `${successRate}%`;
                strategyElement.querySelector('.progress-bar').textContent = `${successRate.toFixed(1)}%`;
            }
        });

        // Update overall stats
        const overallSuccessRate = document.getElementById('overallSuccessRate');
        if (overallSuccessRate) {
            overallSuccessRate.textContent = `${(stats.overall_success_rate * 100).toFixed(2)}%`;
        }
    }

    startPolling() {
        this.pollingInterval = setInterval(() => {
            this.refreshData();
        }, 30000); // Refresh every 30 seconds
    }

    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
    }
}

// Utility functions for formatting
function formatNumber(num) {
    return new Intl.NumberFormat('pt-BR').format(num);
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('pt-BR', {
        style: 'currency',
        currency: 'BRL'
    }).format(amount);
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('pt-BR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    }).format(date);
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new Dashboard();
});

// Make functions available globally
window.formatNumber = formatNumber;
window.formatCurrency = formatCurrency;
window.formatDate = formatDate;

// Show/hide loading states
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<div class="d-flex justify-content-center"><div class="spinner-border" role="status"><span class="visually-hidden">Carregando...</span></div></div>';
    }
}

function hideLoading(elementId, content) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = content;
    }
}

// Toast notifications
function showToast(message, type = 'info') {
    const toastHtml = `
        <div class="toast align-items-center text-bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastElement = toastContainer.lastElementChild;
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    
    // Remove toast from DOM after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

// Export globally
window.showToast = showToast;
window.showLoading = showLoading;
window.hideLoading = hideLoading;
