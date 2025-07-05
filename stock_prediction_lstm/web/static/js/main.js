// Enhanced JavaScript for Stock AI Pro

document.addEventListener('DOMContentLoaded', function() {
    // Initialize enhanced interactivity
    initializeEnhancedUI();
    initializeFormHandling();
    initializePopularStocks();
    initializeTooltips();
    initializeAnimations();
    initializeChartTabs();
});

// Progress simulation variables
let currentProgress = 0;
let currentStep = 0;

// Progress simulation steps
const progressSteps = [
    { text: "Initializing AI models...", duration: 1000 },
    { text: "Fetching market data...", duration: 1500 },
    { text: "Processing historical data...", duration: 2000 },
    { text: "Running sentiment analysis...", duration: 1800 },
    { text: "Training LSTM model...", duration: 2500 },
    { text: "Generating predictions...", duration: 1500 },
    { text: "Creating visualizations...", duration: 1200 },
    { text: "Finalizing analysis...", duration: 800 }
];

function startProgressSimulation() {
    currentProgress = 0;
    currentStep = 0;
    
    const progressCircle = document.getElementById('progress-circle');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressStepText = document.getElementById('progress-status');
    const stepIndicators = document.querySelectorAll('[id^="step-"]');
    
    if (!progressCircle || !progressPercentage || !progressStepText) {
        console.warn('Progress elements not found', {
            progressCircle: !!progressCircle,
            progressPercentage: !!progressPercentage,
            progressStepText: !!progressStepText
        });
        return;
    }
    
    // Initialize step indicators
    stepIndicators.forEach((indicator, index) => {
        indicator.classList.remove('animate-pulse', 'bg-cyan-500', 'bg-emerald-500');
        indicator.classList.add('bg-neutral-600');
    });
    
    // Start the progress animation
    animateProgress();
}

function animateProgress() {
    const progressCircle = document.getElementById('progress-circle');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressStepText = document.getElementById('progress-status');
    const stepIndicators = document.querySelectorAll('[id^="step-"]');
    
    if (currentStep < progressSteps.length) {
        const step = progressSteps[currentStep];
        const targetProgress = ((currentStep + 1) / progressSteps.length) * 100;
        
        // Update step text
        if (progressStepText) {
            progressStepText.textContent = step.text;
            progressStepText.classList.add('fade-in');
        }
        
        // Update step indicators
        if (stepIndicators[currentStep]) {
            stepIndicators[currentStep].classList.add('animate-pulse');
            stepIndicators[currentStep].classList.remove('bg-neutral-600');
            stepIndicators[currentStep].classList.add('bg-cyan-500');
        }
        
        // Animate progress circle and percentage
        const progressDuration = step.duration;
        const progressIncrement = (targetProgress - currentProgress) / (progressDuration / 50);
        
        const progressAnimationInterval = setInterval(() => {
            currentProgress += progressIncrement;
            
            if (currentProgress >= targetProgress) {
                currentProgress = targetProgress;
                clearInterval(progressAnimationInterval);
                
                // Mark current step as completed
                if (stepIndicators[currentStep]) {
                    stepIndicators[currentStep].classList.remove('animate-pulse', 'bg-cyan-500');
                    stepIndicators[currentStep].classList.add('bg-emerald-500');
                }
                
                // Move to next step
                currentStep++;
                
                // Continue with next step or finish
                if (currentStep < progressSteps.length) {
                    // Activate next step indicator
                    if (stepIndicators[currentStep]) {
                        stepIndicators[currentStep].classList.add('animate-pulse');
                        stepIndicators[currentStep].classList.remove('bg-neutral-600');
                        stepIndicators[currentStep].classList.add('bg-cyan-500');
                    }
                    setTimeout(() => animateProgress(), 300);
                } else {
                    // Simulation complete
                    setTimeout(() => {
                        currentProgress = 100;
                        updateProgressDisplay();
                    }, 200);
                }
            }
            
            updateProgressDisplay();
        }, 50);
    }
}

function updateProgressDisplay() {
    const progressCircle = document.getElementById('progress-circle');
    const progressPercentage = document.getElementById('progress-percentage');
    
    if (progressCircle) {
        const circumference = 251.33; // Match the stroke-dasharray value in HTML
        const strokeDashoffset = circumference - (currentProgress / 100) * circumference;
        progressCircle.style.strokeDashoffset = strokeDashoffset;
    }
    
    if (progressPercentage) {
        progressPercentage.textContent = Math.round(currentProgress) + '%';
    }
}

function resetProgress() {
    currentProgress = 0;
    currentStep = 0;
    
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    
    const progressCircle = document.getElementById('progress-circle');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressStepText = document.getElementById('progress-status');
    const stepIndicators = document.querySelectorAll('[id^="step-"]');
    
    if (progressCircle) {
        const circumference = 251.33; // Match the stroke-dasharray value in HTML
        progressCircle.style.strokeDashoffset = circumference;
    }
    
    if (progressPercentage) {
        progressPercentage.textContent = '0%';
    }
    
    if (progressStepText) {
        progressStepText.textContent = 'Preparing analysis...';
    }
    
    stepIndicators.forEach(indicator => {
        indicator.classList.remove('animate-pulse', 'bg-cyan-500', 'bg-emerald-500');
        indicator.classList.add('bg-neutral-600');
    });
}

function showErrorState(errorMessage) {
    const loadingIndicator = document.getElementById('loading-indicator');
    const progressStepText = document.getElementById('progress-status');
    const progressPercentage = document.getElementById('progress-percentage');
    const stepIndicators = document.querySelectorAll('[id^="step-"]');
    
    if (progressStepText) {
        progressStepText.textContent = errorMessage || 'Analysis failed';
        progressStepText.classList.add('text-red-400');
    }
    
    if (progressPercentage) {
        progressPercentage.textContent = 'Error';
        progressPercentage.classList.add('text-red-400');
    }
    
    // Mark all steps as failed
    stepIndicators.forEach(indicator => {
        indicator.classList.remove('animate-pulse', 'bg-cyan-500', 'bg-emerald-500', 'bg-neutral-600');
        indicator.classList.add('bg-red-500');
    });
    
    // Hide loading after a delay
    setTimeout(() => {
        if (loadingIndicator) {
            loadingIndicator.classList.add('hidden');
        }
        showErrorNotification('Analysis failed. Please try again.');
    }, 2000);
}

function initializeEnhancedUI() {
    // Range slider for prediction days
    const predictionDaysSlider = document.getElementById('prediction_days');
    const daysPredictValue = document.getElementById('days-predict-value');
    
    if (predictionDaysSlider && daysPredictValue) {
        predictionDaysSlider.addEventListener('input', function() {
            daysPredictValue.textContent = this.value;
            
            // Add visual feedback
            const percentage = (this.value - this.min) / (this.max - this.min) * 100;
            this.style.background = `linear-gradient(to right, #06b6d4 0%, #06b6d4 ${percentage}%, #374151 ${percentage}%, #374151 100%)`;
        });
        
        // Initialize slider background
        const initialPercentage = (predictionDaysSlider.value - predictionDaysSlider.min) / (predictionDaysSlider.max - predictionDaysSlider.min) * 100;
        predictionDaysSlider.style.background = `linear-gradient(to right, #06b6d4 0%, #06b6d4 ${initialPercentage}%, #374151 ${initialPercentage}%, #374151 100%)`;
    }
    
    // Enhanced ticker input
    const tickerInput = document.getElementById('ticker_symbol');
    if (tickerInput) {
        tickerInput.addEventListener('input', function(e) {
            // Convert to uppercase
            e.target.value = e.target.value.toUpperCase();
            
            // Add typing animation effect
            e.target.classList.add('scale-in');
            setTimeout(() => {
                e.target.classList.remove('scale-in');
            }, 200);
        });
        
        tickerInput.addEventListener('focus', function() {
            this.parentElement.classList.add('ring-2', 'ring-cyan-500', 'ring-opacity-50');
        });
        
        tickerInput.addEventListener('blur', function() {
            this.parentElement.classList.remove('ring-2', 'ring-cyan-500', 'ring-opacity-50');
        });
    }
}

function initializeFormHandling() {
    const analysisForm = document.getElementById('analysis-form');
    const loadingIndicator = document.getElementById('loading-indicator');
    const welcomeScreen = document.getElementById('welcome-screen');
    const runButton = document.getElementById('run-analysis-btn');
    
    if (analysisForm) {
        analysisForm.addEventListener('submit', function(e) {
            // Show enhanced loading state
            showEnhancedLoadingState();
            
            // Add form validation
            const ticker = document.getElementById('ticker_symbol').value.trim();
            if (!ticker) {
                e.preventDefault();
                showErrorNotification('Please enter a stock ticker symbol');
                hideLoadingState();
                return;
            }
            
            if (ticker.length > 5) {
                e.preventDefault();
                showErrorNotification('Ticker symbol should be 5 characters or less');
                hideLoadingState();
                return;
            }
            
            // Smooth scroll to top
            window.scrollTo({ 
                top: 0, 
                behavior: 'smooth' 
            });
        });
    }
    
    function showEnhancedLoadingState() {
        if (welcomeScreen) {
            welcomeScreen.style.opacity = '0';
            welcomeScreen.style.transform = 'translateY(-20px)';
            setTimeout(() => {
                welcomeScreen.classList.add('hidden');
            }, 300);
        }
        
        if (loadingIndicator) {
            loadingIndicator.classList.remove('hidden');
            loadingIndicator.style.opacity = '0';
            loadingIndicator.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                loadingIndicator.style.opacity = '1';
                loadingIndicator.style.transform = 'translateY(0)';
                
                // Start progress simulation
                startProgressSimulation();
            }, 100);
        }
        
        if (runButton) {
            runButton.disabled = true;
            runButton.innerHTML = `
                <span class="flex items-center justify-center">
                    <svg class="animate-spin w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                    </svg>
                    Processing...
                </span>
            `;
        }
    }
    
    function hideLoadingState() {
        // Reset progress simulation
        resetProgress();
        
        if (loadingIndicator) {
            loadingIndicator.classList.add('hidden');
        }
        
        if (welcomeScreen) {
            welcomeScreen.classList.remove('hidden');
            welcomeScreen.style.opacity = '1';
            welcomeScreen.style.transform = 'translateY(0)';
        }
        
        if (runButton) {
            runButton.disabled = false;
            runButton.innerHTML = `
                <span class="flex items-center justify-center">
                    <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                    </svg>
                    Run AI Analysis
                </span>
            `;
        }
    }
}

function initializePopularStocks() {
    const popularStockBtns = document.querySelectorAll('.popular-stock-btn');
    const tickerInput = document.getElementById('ticker_symbol');
    
    popularStockBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const ticker = this.dataset.ticker;
            
            if (tickerInput) {
                // Add selection animation
                tickerInput.value = ticker;
                tickerInput.classList.add('scale-in');
                
                // Visual feedback for selected stock
                popularStockBtns.forEach(b => {
                    b.classList.remove('bg-gradient-to-r', 'from-cyan-500/20', 'to-blue-500/20', 'border-cyan-500');
                    b.classList.add('bg-neutral-800/50', 'border-neutral-700/50');
                });
                
                this.classList.remove('bg-neutral-800/50', 'border-neutral-700/50');
                this.classList.add('bg-gradient-to-r', 'from-cyan-500/20', 'to-blue-500/20', 'border-cyan-500');
                
                // Remove animation class after animation completes
                setTimeout(() => {
                    tickerInput.classList.remove('scale-in');
                }, 200);
                
                // Show success notification
                showSuccessNotification(`Selected ${ticker}`);
            }
        });
        
        // Add hover effects
        btn.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        btn.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

function initializeTooltips() {
    // Add tooltips to various elements
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            showTooltip(this, this.dataset.tooltip);
        });
        
        element.addEventListener('mouseleave', function() {
            hideTooltip();
        });
    });
}

function initializeAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe elements that should animate on scroll
    const animatableElements = document.querySelectorAll('.metric-card-enhanced, .chart-container, .feature-card');
    animatableElements.forEach(el => observer.observe(el));
    
    // Add staggered animation to feature cards
    const featureCards = document.querySelectorAll('.group');
    featureCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
}

function initializeChartTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    if (tabButtons.length === 0) return;

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and hide all content
            tabButtons.forEach(btn => {
                btn.classList.remove('active-tab', 'border-cyan-500', 'text-cyan-400');
                btn.classList.add('border-transparent', 'text-neutral-400');
            });
            tabContents.forEach(content => content.classList.add('hidden'));

            // Add active class to clicked button and show corresponding content
            button.classList.add('active-tab', 'border-cyan-500', 'text-cyan-400');
            button.classList.remove('border-transparent', 'text-neutral-400');
            
            const targetTab = document.getElementById(button.dataset.tab);
            if (targetTab) {
                targetTab.classList.remove('hidden');
                targetTab.classList.add('fade-in');
            }
        });
    });

    // Set initial active tab
    const firstTab = document.querySelector('.tab-button.active-tab');
    if (firstTab) {
        firstTab.click();
    }
}

// Utility functions
function showSuccessNotification(message) {
    showNotification(message, 'success');
}

function showErrorNotification(message) {
    showNotification(message, 'error');
}

function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => notification.remove());
    
    const notification = document.createElement('div');
    notification.classList.add('notification', 'fixed', 'top-4', 'right-4', 'z-50', 'p-4', 'rounded-lg', 'shadow-lg', 'transition-all', 'duration-300', 'transform', 'translate-x-full');
    
    const colors = {
        success: 'bg-emerald-500 text-white',
        error: 'bg-red-500 text-white',
        info: 'bg-cyan-500 text-white'
    };
    
    notification.classList.add(...colors[type].split(' '));
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.remove('translate-x-full');
    }, 100);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.classList.add('translate-x-full');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

function showTooltip(element, text) {
    const tooltip = document.createElement('div');
    tooltip.classList.add('tooltip-popup', 'absolute', 'z-50', 'px-3', 'py-2', 'text-sm', 'text-white', 'bg-gray-900', 'rounded-lg', 'shadow-lg', 'opacity-0', 'transition-opacity', 'duration-200');
    tooltip.textContent = text;
    
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
    
    setTimeout(() => {
        tooltip.classList.remove('opacity-0');
    }, 100);
}

function hideTooltip() {
    const tooltip = document.querySelector('.tooltip-popup');
    if (tooltip) {
        tooltip.classList.add('opacity-0');
        setTimeout(() => {
            tooltip.remove();
        }, 200);
    }
}

// Accordion functionality
function toggleAccordion(id) {
    const content = document.getElementById(id + '-content');
    const arrow = document.getElementById(id + '-arrow');
    
    if (content && arrow) {
        const isHidden = content.classList.contains('hidden');
        
        if (isHidden) {
            content.classList.remove('hidden');
            content.style.maxHeight = '0';
            content.style.opacity = '0';
            
            // Animate expansion
            setTimeout(() => {
                content.style.maxHeight = content.scrollHeight + 'px';
                content.style.opacity = '1';
            }, 10);
            
            arrow.style.transform = 'rotate(180deg)';
        } else {
            content.style.maxHeight = '0';
            content.style.opacity = '0';
            arrow.style.transform = 'rotate(0deg)';
            
            setTimeout(() => {
                content.classList.add('hidden');
            }, 300);
        }
    }
}

// Tab functionality for dashboard pages
function switchTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('[data-tab-content]');
    const tabButtons = document.querySelectorAll('[data-tab]');
    
    tabContents.forEach(content => {
        content.classList.add('hidden');
        content.classList.remove('fade-in');
    });
    
    tabButtons.forEach(button => {
        button.classList.remove('active-tab');
    });
    
    // Show selected tab content
    const targetContent = document.querySelector(`[data-tab-content="${tabName}"]`);
    const targetButton = document.querySelector(`[data-tab="${tabName}"]`);
    
    if (targetContent) {
        targetContent.classList.remove('hidden');
        setTimeout(() => {
            targetContent.classList.add('fade-in');
        }, 50);
    }
    
    if (targetButton) {
        targetButton.classList.add('active-tab');
    }
}

// Enhanced chart responsiveness
function handleChartResize() {
    const charts = document.querySelectorAll('.chart-container');
    charts.forEach(chart => {
        // Trigger chart library resize if available
        if (window.Plotly && chart.querySelector('.plotly-graph-div')) {
            window.Plotly.Plots.resize(chart.querySelector('.plotly-graph-div'));
        }
    });
}

// Listen for window resize
window.addEventListener('resize', debounce(handleChartResize, 250));

// Debounce utility function
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

// Make functions globally available
window.toggleAccordion = toggleAccordion;
window.switchTab = switchTab;
window.formatCurrency = formatCurrency;
window.formatPercentage = formatPercentage;
window.showErrorState = showErrorState;
