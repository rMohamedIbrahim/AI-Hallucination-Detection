document.addEventListener('DOMContentLoaded', () => {
    console.log("TruthGuard JavaScript loaded");
    
    // Animate result cards
    const resultCards = document.querySelectorAll('.result-cards > div');
    resultCards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('show');
            card.style.animation = `fadeInUp 0.5s ease ${index * 0.1}s forwards`;
        }, index * 50);
    });

    // Form submission with loading animation
    const form = document.getElementById('promptForm');
    const submitBtn = document.getElementById('submitBtn');
    const voiceBtn = document.getElementById('startVoiceBtn');
    const submitSpinner = submitBtn?.querySelector('.spinner-border');
    const voiceSpinner = voiceBtn?.querySelector('.spinner-border');
    if (form && submitBtn && submitSpinner) {
        form.addEventListener('submit', (e) => {
            console.log("Form submitted");
            submitBtn.disabled = true;
            submitBtn.classList.add('loading');
            submitSpinner.classList.remove('d-none');
            
            // Show loading overlay
            const loadingOverlay = document.createElement('div');
            loadingOverlay.className = 'loading-overlay';
            const spinner = document.createElement('div');
            spinner.className = 'loading-spinner';
            loadingOverlay.appendChild(spinner);
            document.body.appendChild(loadingOverlay);
            
            // Force rendering before adding active class for transition
            setTimeout(() => {
                loadingOverlay.classList.add('active');
            }, 10);
            
            // Remove overlay on navigation or after timeout
            const removeOverlay = () => {
                loadingOverlay.classList.remove('active');
                setTimeout(() => {
                    loadingOverlay.remove();
                }, 300);
            };
            
            // Set timeout for long-running requests
            setTimeout(() => {
                if (document.body.contains(loadingOverlay)) {
                    removeOverlay();
                }
            }, 15000);
        });
    } else {
        console.error("Form elements not found:", { form, submitBtn, submitSpinner });
    }

    // Smooth scroll to results
    if (resultCards.length > 0) {
        setTimeout(() => {
            const resultsContainer = document.querySelector('.results-container');
            if (resultsContainer) {
                resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 500);
    }

    // Initialize tab functionality
    initializeTabs();
    
    // Initialize charts if results are available
    initializeCharts();
    
    // Voice input functionality
    if (voiceBtn && voiceSpinner) {
        voiceBtn.addEventListener('click', startVoiceInput);
    }
});

// Tabs functionality
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Add active class to current button
            button.classList.add('active');
            
            // Show corresponding pane
            const tabId = button.getAttribute('data-tab');
            const pane = document.getElementById(tabId + '-tab');
            if (pane) {
                pane.classList.add('active');
            }
        });
    });
}

// Initialize charts
function initializeCharts() {
    // Confidence pie chart
    const confidencePieChart = document.getElementById('confidencePieChart');
    if (confidencePieChart) {
        const googleConfidence = parseFloat(document.getElementById('confidenceGoogle')?.value || 0);
        const groqConfidence = parseFloat(document.getElementById('confidenceGroq')?.value || 0);
        const cohereConfidence = parseFloat(document.getElementById('confidenceCohere')?.value || 0);
        
        new Chart(confidencePieChart, {
            type: 'pie',
            data: {
                labels: ['Google', 'Groq', 'Cohere'],
                datasets: [{
                    data: [googleConfidence, groqConfidence, cohereConfidence],
                    backgroundColor: ['#4285F4', '#FF9900', '#5BC2A8'],
                    borderColor: ['#3367D6', '#E88C00', '#4AA48D'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: {
                                family: 'Inter, sans-serif',
                                size: 12
                            },
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                return `${context.label}: ${(value * 100).toFixed(0)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Hallucination radar chart
    const hallucinationRadarChart = document.getElementById('hallucinationRadarChart');
    if (hallucinationRadarChart) {
        const googleHallucination = parseFloat(document.getElementById('hallucinationGoogle')?.value || 0);
        const groqHallucination = parseFloat(document.getElementById('hallucinationGroq')?.value || 0);
        const cohereHallucination = parseFloat(document.getElementById('hallucinationCohere')?.value || 0);
        
        new Chart(hallucinationRadarChart, {
            type: 'radar',
            data: {
                labels: ['Factual Accuracy', 'Source Verification', 'Logical Consistency'],
                datasets: [
                    {
                        label: 'Google',
                        data: [1 - googleHallucination, 1 - googleHallucination * 0.8, 1 - googleHallucination * 1.2],
                        backgroundColor: 'rgba(66, 133, 244, 0.2)',
                        borderColor: '#4285F4',
                        pointBackgroundColor: '#4285F4',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#4285F4'
                    },
                    {
                        label: 'Groq',
                        data: [1 - groqHallucination, 1 - groqHallucination * 0.9, 1 - groqHallucination * 1.1],
                        backgroundColor: 'rgba(255, 153, 0, 0.2)',
                        borderColor: '#FF9900',
                        pointBackgroundColor: '#FF9900',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#FF9900'
                    },
                    {
                        label: 'Cohere',
                        data: [1 - cohereHallucination, 1 - cohereHallucination * 0.85, 1 - cohereHallucination * 1.15],
                        backgroundColor: 'rgba(91, 194, 168, 0.2)',
                        borderColor: '#5BC2A8',
                        pointBackgroundColor: '#5BC2A8',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#5BC2A8'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            stepSize: 0.2,
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        },
                        pointLabels: {
                            font: {
                                family: 'Inter, sans-serif',
                                size: 12
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: {
                                family: 'Inter, sans-serif',
                                size: 12
                            },
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                return `${context.dataset.label}: ${(value * 100).toFixed(0)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Hallucination bar chart
    const hallucinationBarChart = document.getElementById('hallucinationBarChart');
    if (hallucinationBarChart) {
        const googleHallucination = parseFloat(document.getElementById('hallucinationGoogle')?.value || 0);
        const groqHallucination = parseFloat(document.getElementById('hallucinationGroq')?.value || 0);
        const cohereHallucination = parseFloat(document.getElementById('hallucinationCohere')?.value || 0);

        new Chart(hallucinationBarChart, {
            type: 'bar',
            data: {
                labels: ['Google', 'Groq', 'Cohere'],
                datasets: [{
                    label: 'Hallucination Score',
                    data: [googleHallucination, groqHallucination, cohereHallucination],
                    backgroundColor: ['rgba(66, 133, 244, 0.5)', 'rgba(255, 153, 0, 0.5)', 'rgba(91, 194, 168, 0.5)'],
                    borderColor: ['#4285F4', '#FF9900', '#5BC2A8'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function (value) {
                                return (value * 100) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                const value = context.raw;
                                return `${context.label}: ${(value * 100).toFixed(0)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
}

// Voice input functionality
function startVoiceInput() {
    const voiceBtn = document.getElementById('startVoiceBtn');
    const voiceSpinner = voiceBtn?.querySelector('.spinner-border');
    const promptTextarea = document.getElementById('prompt');
    const statusIndicator = document.getElementById('status');
    
    if (voiceBtn && voiceSpinner && promptTextarea && statusIndicator) {
        // Check if browser supports speech recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SpeechRecognition();
            
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            // Start recording
            voiceBtn.disabled = true;
            voiceBtn.classList.add('loading');
            voiceSpinner.classList.remove('d-none');
            statusIndicator.innerHTML = '<i class="fas fa-microphone-alt recording"></i><span>Listening...</span>';
            
            recognition.start();
            
            let finalTranscript = '';
            
            recognition.onresult = (event) => {
                let interimTranscript = '';
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript + ' ';
                    } else {
                        interimTranscript += transcript;
                    }
                }
                
                // Update the textarea with current transcription
                promptTextarea.value = finalTranscript + interimTranscript;
            };
            
            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                
                // Reset UI
                voiceBtn.disabled = false;
                voiceBtn.classList.remove('loading');
                voiceSpinner.classList.add('d-none');
                statusIndicator.innerHTML = '<i class="fas fa-circle error"></i><span>Error with speech recognition</span>';
                
                setTimeout(() => {
                    statusIndicator.innerHTML = '<i class="fas fa-circle ready"></i><span>Ready for verification</span>';
                }, 3000);
            };
            
            recognition.onend = () => {
                // Reset UI
                voiceBtn.disabled = false;
                voiceBtn.classList.remove('loading');
                voiceSpinner.classList.add('d-none');
                statusIndicator.innerHTML = '<i class="fas fa-check-circle"></i><span>Voice input completed</span>';
                
                setTimeout(() => {
                    statusIndicator.innerHTML = '<i class="fas fa-circle ready"></i><span>Ready for verification</span>';
                }, 3000);
            };
        } else {
            // Browser doesn't support speech recognition
            alert('Speech recognition is not supported in your browser. Please try using Chrome.');
            
            // Reset UI
            voiceBtn.disabled = false;
            voiceBtn.classList.remove('loading');
            voiceSpinner.classList.add('d-none');
        }
    }
}

// Highlight hallucination markers
function highlightHallucinations() {
    const hallucinations = document.querySelectorAll('.hallucination-marker');
    
    hallucinations.forEach(marker => {
        marker.addEventListener('mouseover', () => {
            marker.classList.add('active');
        });
        
        marker.addEventListener('mouseout', () => {
            marker.classList.remove('active');
        });
    });
}

// Add resize observer for charts
window.addEventListener('resize', () => {
    // Chart.js 4.x+ uses Chart.getChart() and Chart.instances is an object, not iterable
    if (window.Chart && Chart.instances) {
        // Chart.js 2.x/3.x: Chart.instances is an object or array
        if (typeof Chart.instances === 'object') {
            Object.values(Chart.instances).forEach(chart => {
                if (chart && typeof chart.resize === 'function') {
                    chart.resize();
                }
            });
        }
    }
});

// Add copy functionality for answers
const addCopyButtons = () => {
    const answerContents = document.querySelectorAll('.answer-content');
    
    answerContents.forEach(content => {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
        copyBtn.title = 'Copy to clipboard';
        
        content.parentNode.style.position = 'relative';
        content.parentNode.appendChild(copyBtn);
        
        copyBtn.addEventListener('click', () => {
            const textToCopy = content.textContent;
            navigator.clipboard.writeText(textToCopy).then(() => {
                copyBtn.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
                }, 2000);
            });
        });
    });
};

// Execute the copy functionality after DOM is fully loaded
document.addEventListener('DOMContentLoaded', addCopyButtons);