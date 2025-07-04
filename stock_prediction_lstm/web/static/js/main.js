// Enhanced JavaScript for Stock AI Dashboard

document.addEventListener('DOMContentLoaded', function() {
    // Initialize chart tabs if they exist
    initializeChartTabs();
    
    // Initialize form enhancements
    initializeFormEnhancements();
    
    // Initialize tooltips and help text
    initializeHelpers();
});

function initializeChartTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    if (tabButtons.length === 0) return;

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and hide all content
            tabButtons.forEach(btn => {
                btn.classList.remove('active-tab', 'border-indigo-500', 'text-indigo-600');
                btn.classList.add('border-transparent', 'text-gray-600');
            });
            tabContents.forEach(content => content.classList.add('hidden'));

            // Add active class to clicked button and show corresponding content
            button.classList.add('active-tab', 'border-indigo-500', 'text-indigo-600');
            button.classList.remove('border-transparent', 'text-gray-600');
            
            const targetTab = document.getElementById(button.dataset.tab);
            if (targetTab) {
                targetTab.classList.remove('hidden');
            }
        });
    });

    // Set initial active tab
    const firstTab = document.querySelector('.tab-button.active-tab');
    if (firstTab) {
        firstTab.click();
    }
}

function initializeFormEnhancements() {
    // Days to predict input enhancement
    const daysInput = document.getElementById('prediction_days');
    const daysDisplay = document.getElementById('days-predict-value');
    
    if (daysInput && daysDisplay) {
        daysInput.addEventListener('input', () => {
            daysDisplay.textContent = daysInput.value;
        });
    }

    // Popular stock selection
    const popularStockBtns = document.querySelectorAll('.popular-stock-btn');
    const tickerInput = document.getElementById('ticker_symbol');
    
    if (popularStockBtns.length > 0 && tickerInput) {
        popularStockBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const ticker = btn.dataset.ticker;
                tickerInput.value = ticker;
                
                // Visual feedback
                popularStockBtns.forEach(b => {
                    b.classList.remove('bg-indigo-100', 'border-indigo-300');
                    b.classList.add('bg-white', 'border-gray-200');
                });
                btn.classList.add('bg-indigo-100', 'border-indigo-300');
                btn.classList.remove('bg-white', 'border-gray-200');
            });
        });
    }

    // Ticker input uppercase conversion
    if (tickerInput) {
        tickerInput.addEventListener('input', (e) => {
            e.target.value = e.target.value.toUpperCase();
        });
    }

    // Form submission enhancement
    const analysisForm = document.getElementById('analysis-form');
    const loadingIndicator = document.getElementById('loading-indicator');
    const welcomeScreen = document.getElementById('welcome-screen');
    
    if (analysisForm && loadingIndicator) {
        analysisForm.addEventListener('submit', (e) => {
            // Show loading indicator
            if (welcomeScreen) {
                welcomeScreen.classList.add('hidden');
            }
            loadingIndicator.classList.remove('hidden');
            
            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }
}

function initializeHelpers() {
    // Add tooltip functionality for info icons
    const infoIcons = document.querySelectorAll('[title]');
    infoIcons.forEach(icon => {
        icon.addEventListener('mouseenter', (e) => {
            // You can add custom tooltip styling here
            e.target.style.cursor = 'help';
        });
    });
}

// Accordion toggle function
function toggleAccordion(id) {
    const content = document.getElementById(id + '-content');
    const arrow = document.getElementById(id + '-arrow');
    
    if (content && arrow) {
        if (content.classList.contains('hidden')) {
            content.classList.remove('hidden');
            arrow.classList.add('rotate-180');
        } else {
            content.classList.add('hidden');
            arrow.classList.remove('rotate-180');
        }
    }
}

// Utility function to format numbers
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}

// Export functions for global use
window.toggleAccordion = toggleAccordion;
window.formatCurrency = formatCurrency;
window.formatPercentage = formatPercentage;
