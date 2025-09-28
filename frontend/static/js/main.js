class PersonalityChat {
    constructor() {
        this.apiBaseUrl = 'http://127.0.0.1:8000';
        this.currentSessionId = null;
        this.userName = "Your";
        this.sentimentChart = null; // NEW: Chart instance

        this.dom = {
            chatWindow: document.getElementById('chat-window'),
            textInput: document.getElementById('text-input'),
            analyzeBtn: document.getElementById('analyze-btn'),
            sidebarTitle: document.getElementById('sidebar-title'),
            traitDetails: document.getElementById('trait-details'),
            apiStatus: document.getElementById('api-status'), // NEW
            summaryBtn: document.getElementById('summary-btn'), // NEW
            newChatBtn: document.getElementById('new-chat-btn'),
            sentimentChartCanvas: document.getElementById('sentiment-chart'), // NEW
            nameModal: document.getElementById('name-modal-overlay'),
            nameInput: document.getElementById('name-input'),
            submitNameBtn: document.getElementById('submit-name-btn'),
            explanationModal: document.getElementById('explanation-modal-overlay'),
            explanationContent: document.getElementById('explanation-modal-content')
        };
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkApiStatus();
        this.initSentimentChart(); // NEW
        this.loadState(); // NEW: Try to load a previous session
    }

    bindEvents() {
        this.dom.analyzeBtn.addEventListener('click', () => this.sendMessage());
        this.dom.textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendMessage(); }
        });
        this.dom.textInput.addEventListener('input', () => this.autoResizeTextarea());

        this.dom.submitNameBtn.addEventListener('click', () => this.handleNameSubmit());
        this.dom.nameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.handleNameSubmit();
        });

        // NEW: Summary button event
        this.dom.summaryBtn.addEventListener('click', () => this.getFinalSummary());
        this.dom.newChatBtn.addEventListener('click', () => this.resetChat());
        // Modal close event
        this.dom.explanationModal.addEventListener('click', (e) => {
            if (e.target === this.dom.explanationModal || e.target.classList.contains('close-modal')) {
                this.dom.explanationModal.classList.remove('active');
            }
        });
    }
    
    // NEW: Load state from localStorage
    async loadState() {
        const savedState = localStorage.getItem('personalityChatSession');
        if (savedState) {
            const { sessionId, userName, chatHTML } = JSON.parse(savedState);
            if (sessionId && userName) {
                this.currentSessionId = sessionId;
                this.userName = userName;
                
                // Restore UI
                this.dom.sidebarTitle.textContent = `${this.userName}'s Profile`;
                this.dom.chatWindow.innerHTML = chatHTML;
                this.dom.nameModal.classList.remove('active');
                this.dom.textInput.disabled = false;
                this.dom.analyzeBtn.disabled = false;
                this.dom.textInput.focus();

                // Fetch latest session data from server to update sidebar/chart
                try {
                    const response = await fetch(`${this.apiBaseUrl}/session/${this.currentSessionId}`);
                    if (!response.ok) throw new Error('Session expired or not found.');
                    const data = await response.json();
                    this.updateTraitDetails(data.trait_descriptions);
                    this.updateSentimentChart(data.sentiment_history);
                } catch (error) {
                    console.error("Failed to restore session from server:", error);
                    // If server session is lost, clear local and start over
                    this.clearState(); 
                    this.dom.nameModal.classList.add('active');
                }
                
                return; // Stop if state was loaded
            }
        }
        // If no saved state, ensure UI is in default start mode
        this.dom.nameModal.classList.add('active');
    }

    // NEW: Save state to localStorage
    saveState() {
        if (this.currentSessionId && this.userName) {
            const state = {
                sessionId: this.currentSessionId,
                userName: this.userName,
                chatHTML: this.dom.chatWindow.innerHTML
            };
            localStorage.setItem('personalityChatSession', JSON.stringify(state));
        }
    }

    // NEW: Clear state
    clearState() {
        localStorage.removeItem('personalityChatSession');
        this.currentSessionId = null;
        this.userName = "Your";
        this.dom.chatWindow.innerHTML = '';
        this.updateTraitDetails({});
        this.updateSentimentChart([]);
    }


    handleNameSubmit() {
        const name = this.dom.nameInput.value.trim();
        if (name) {
            this.userName = name;
            this.dom.sidebarTitle.textContent = `${this.userName}'s Profile`;
            this.dom.nameModal.classList.remove('active');
            this.dom.textInput.disabled = false;
            this.dom.analyzeBtn.disabled = false;
            this.dom.textInput.focus();
            this.addMessageToChat(`Hello, ${this.userName}! To begin, please tell me a bit about yourself. You could describe what you enjoy, how you handle challenges, or what a typical day looks like for you.`, 'bot');
        }
    }

    autoResizeTextarea() {
        this.dom.textInput.style.height = 'auto';
        this.dom.textInput.style.height = `${this.dom.textInput.scrollHeight}px`;
    }

    addMessageToChat(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        const avatarIcon = sender === 'user' ? 'fa-user' : 'fa-brain'; // Changed bot icon
        const avatarClass = sender === 'user' ? 'user-avatar' : 'bot-avatar';
        messageElement.innerHTML = `<div class="avatar ${avatarClass}"><i class="fa-solid ${avatarIcon}"></i></div><div class="message-content">${text}</div>`;
        this.dom.chatWindow.appendChild(messageElement);
        this.dom.chatWindow.scrollTop = this.dom.chatWindow.scrollHeight;
    }

    async sendMessage() {
        const text = this.dom.textInput.value.trim();
        if (!text) return;
        this.addMessageToChat(text, 'user');
        this.dom.textInput.value = '';
        this.autoResizeTextarea();
        this.dom.analyzeBtn.disabled = true;

        try {
            const response = await fetch(`${this.apiBaseUrl}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, session_id: this.currentSessionId })
            });
            if (!response.ok) throw new Error(`API error: ${response.statusText}`);
            const data = await response.json();

            this.currentSessionId = data.session_id;

            if (data.followup_question) {
                this.addMessageToChat(data.followup_question, 'bot');
            }
            if (data.ocean_scores) {
                this.updateTraitDetails(data.trait_descriptions);
            }
            // NEW: Update chart with new history
            if(data.sentiment_history) {
                this.updateSentimentChart(data.sentiment_history);
            }

            this.saveState(); // NEW: Save state after successful interaction
        } catch (error) {
            console.error("Send message error:", error);
            this.addMessageToChat('Sorry, I encountered an error. Please check the connection and try again.', 'bot');
        } finally {
            this.dom.analyzeBtn.disabled = false;
            this.dom.textInput.focus();
        }
    }

    updateTraitDetails(traitDescriptions) {
        this.dom.traitDetails.innerHTML = '';
        const traits = (traitDescriptions && Object.keys(traitDescriptions).length > 0) ? traitDescriptions : {
            O: { name: 'Openness', score: 0, level: 'N/A' },
            C: { name: 'Conscientiousness', score: 0, level: 'N/A' },
            E: { name: 'Extraversion', score: 0, level: 'N/A' },
            A: { name: 'Agreeableness', score: 0, level: 'N/A' },
            N: { name: 'Neuroticism', score: 0, level: 'N/A' }
        };
        
        // NEW: Enable/disable summary button based on scores
        const hasScores = Object.values(traits).some(t => t.score > 0);
        this.dom.summaryBtn.disabled = !hasScores;


        Object.values(traits).forEach(info => {
            const card = document.createElement('div');
            card.className = 'trait-card';
            const score = info.score;
            let barColor = 'var(--secondary-color)'; // Moderate
            if (score < 40) barColor = 'var(--warning-color)';
            if (score > 60) barColor = 'var(--success-color)';
            
            card.innerHTML = `<div class="trait-header"><span class="trait-name">${info.name}</span><span class="trait-score">${score}%</span></div><div class="progress-bar"><div class="progress-bar-fill" style="width: ${score}%; background-color: ${barColor};"></div></div>`;
            
            if (score > 0) {
                card.addEventListener('click', () => this.getTraitExplanation(info.name, score));
            } else {
                card.style.cursor = 'default';
                card.style.opacity = '0.6';
            }
            this.dom.traitDetails.appendChild(card);
        });
    }

    async getTraitExplanation(traitName, score) {
        this.showModalSpinner(`Generating personalized insights for <strong>${traitName}</strong>...`);
        try {
            const response = await fetch(`${this.apiBaseUrl}/explain-trait`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ trait_name: traitName, score: score })
            });
            if (!response.ok) throw new Error('Failed to fetch explanation.');
            const data = await response.json();
            this.dom.explanationContent.innerHTML = `<h3>${traitName} (${score}%)</h3><p>${data.explanation}</p>`;
        } catch (error) {
            this.dom.explanationContent.innerHTML = `<h3>Error</h3><p>Could not load the explanation at this time. Please try again later.</p>`;
        }
    }
    
    // NEW: Function to get the final summary
    async getFinalSummary() {
        if (!this.currentSessionId) return;
        this.showModalSpinner("Analyzing your conversation to create a final summary...");
        try {
            const response = await fetch(`${this.apiBaseUrl}/summary/${this.currentSessionId}`);
            if (!response.ok) throw new Error('Failed to fetch summary.');
            const data = await response.json();
            this.dom.explanationContent.innerHTML = `<h3>Your Personality Summary</h3><p>${data.summary}</p>`;
        } catch (error) {
            this.dom.explanationContent.innerHTML = `<h3>Error</h3><p>Could not load the summary at this time. Please try again later.</p>`;
        }
    }

    // NEW: Helper to show modal with loading state
    showModalSpinner(message) {
        this.dom.explanationContent.innerHTML = `<div class="loading-spinner"><i class="fas fa-brain fa-spin"></i><p>${message}</p></div>`;
        this.dom.explanationModal.classList.add('active');
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (!response.ok) throw new Error('API not healthy');
            const data = await response.json();
            if (data.status === 'healthy' && data.model_loaded) {
                this.dom.apiStatus.classList.add('online');
                this.dom.apiStatus.classList.remove('offline');
                this.dom.apiStatus.querySelector('span:last-child').textContent = 'API Online';
            } else {
                throw new Error('API reported an issue.');
            }
        } catch (error) {
            this.dom.apiStatus.classList.add('offline');
            this.dom.apiStatus.classList.remove('online');
            this.dom.apiStatus.querySelector('span:last-child').textContent = 'API Offline';
            console.error("API Status Check Failed:", error);
        }
    }

    // NEW: Initialize the sentiment chart
    // NEW: Initialize the sentiment chart
    initSentimentChart() {
        const ctx = this.dom.sentimentChartCanvas.getContext('2d');
        this.sentimentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Sentiment Trend',
                    data: [],
                    borderColor: 'rgba(155, 89, 182, 0.8)',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: 'rgba(155, 89, 182, 1)',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false, // ESSENTIAL for filling the container
                layout: {
                    padding: {
                        left: 25,
                        right: 10,
                        top: 5,
                        bottom: 15
                    }
                },
                scales: {
                    y: {
                        min: -1,
                        max: 1,
                        ticks: {
                            stepSize: 1, // GUARANTEES ticks are at -1, 0, and 1
                            callback: function(value) {
                                if (value === 1) return 'Positive';
                                if (value === 0) return 'Neutral';
                                if (value === -1) return 'Negative';
                                return null;
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    x: {
                        display: false,
                        grid: {
                            offset: true // This is the new fix
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        xAlign: 'left', // Aligns the tooltip to the right of the point
                        yAlign: 'bottom'  // Keeps the tooltip below the point
                    }
                }
            }
        });
        this.updateSentimentChart([]); // Initialize empty
    }

    // NEW: Update the chart with new data
    updateSentimentChart(sentimentHistory) {
    if (!this.sentimentChart) return;

    // NEW: Add/remove a class based on whether the chart has data
    if (sentimentHistory.length === 0) {
        this.dom.sentimentChartCanvas.parentElement.classList.add('is-empty');
    } else {
        this.dom.sentimentChartCanvas.parentElement.classList.remove('is-empty');
    }
    
    const sentimentMap = { 'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1 };
    
    this.sentimentChart.data.labels = sentimentHistory.map((_, i) => `Turn ${i + 1}`);
    this.sentimentChart.data.datasets[0].data = sentimentHistory.map(s => sentimentMap[s] || 0);
    
    this.sentimentChart.update('none'); 
}

    async resetChat() {
        if (!this.currentSessionId || !confirm("Are you sure you want to end this session and start a new one?")) {
            // If there's no session yet, just reload
            if(!this.currentSessionId) location.reload();
            return;
        }

        try {
            await fetch(`${this.apiBaseUrl}/reset/${this.currentSessionId}`, { method: 'POST' });
        } catch (error) {
            console.error("Failed to reset session on server, clearing client-side anyway.", error);
        } finally {
            // Clear local storage and reload the page to its original state
            localStorage.removeItem('personalityChatSession');
            location.reload();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new PersonalityChat();
});