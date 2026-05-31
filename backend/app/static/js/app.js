import { appState } from './state.js';
import { checkSystemStatus, loadDashboardStats, fetchDBFacets, runEvaluation, clearAllHistory } from './api.js';
import { showToast, filterEvaluationFacets, filterEvalCategory, fetchAndRenderDBFacets, renderResultsMetadata, renderFacetScores } from './ui.js';

document.addEventListener("DOMContentLoaded", () => {
    checkSystemStatus();
    loadDashboardStats();
    fetchDBFacets();
    
    setInterval(checkSystemStatus, 15000);
    setInterval(loadDashboardStats, 30000);
});

function switchTab(tabId) {
    appState.activeTab = tabId;
    
    document.querySelectorAll('.side-nav .nav-item').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
    
    const targetBtn = document.querySelector(`.side-nav button[onclick="switchTab('${tabId}')"]`);
    if (targetBtn) targetBtn.classList.add('active');
    
    const targetPane = document.getElementById(`tab-${tabId}`);
    if (targetPane) targetPane.classList.add('active');
    
    const title = document.getElementById('page-title');
    const subtitle = document.getElementById('page-subtitle');
    
    if (tabId === 'evaluator') {
        title.innerText = 'Turn Evaluator';
        subtitle.innerText = 'Analyze conversation turns against 300+ evaluation dimensions instantly.';
    } else if (tabId === 'facets-db') {
        title.innerText = 'Facet Database';
        subtitle.innerText = 'Browse and query loaded assessment criteria stored in MongoDB.';
        fetchAndRenderDBFacets();
    } else if (tabId === 'history') {
        title.innerText = 'Evaluation History';
        subtitle.innerText = 'Reload and compare past conversation evaluations.';
        loadHistory();
    }
}

async function loadHistory() {
    const container = document.getElementById("history-container");
    container.innerHTML = `<div style="text-align: center; padding: 40px;"><i class="fa-solid fa-spinner fa-spin"></i> Fetching evaluation logs...</div>`;
    
    try {
        const response = await fetch('/api/history?limit=50');
        if (!response.ok) throw new Error();
        
        const history = await response.json();
        
        container.innerHTML = "";
        if (history.length === 0) {
            container.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-muted);"><i class="fa-solid fa-folder-open" style="font-size: 32px; margin-bottom: 12px; display: block;"></i> No evaluation logs found. Make your first evaluation on the left panel!</div>`;
            return;
        }
        
        history.forEach(item => {
            const dateStr = new Date(item.created_at).toLocaleString();
            const rawScoreCount = Object.keys(item.facet_scores).length;
            
            const card = document.createElement("div");
            card.className = "history-item";
            
            const userSnippet = item.user.length > 90 ? item.user.substring(0, 90) + "..." : item.user;
            const assistantSnippet = item.assistant.length > 90 ? item.assistant.substring(0, 90) + "..." : item.assistant;
            
            card.innerHTML = `
                <div class="history-top">
                    <span class="history-title">Convo: ${item.conversation_id} (Turn ${item.turn_id})</span>
                    <span class="history-time">${dateStr}</span>
                </div>
                <div class="history-dialogue">
                    <div class="dialogue-row"><strong>User:</strong> "${userSnippet}"</div>
                    <div class="dialogue-row"><strong>AI:</strong> "${assistantSnippet}"</div>
                </div>
                <div class="history-badges">
                    <span class="history-pill"><i class="fa-solid fa-folder"></i> Topic: ${item.metadata.topic}</span>
                    <span class="history-pill"><i class="fa-solid fa-bullseye"></i> Intent: ${item.metadata.intent}</span>
                    <span class="history-pill"><i class="fa-solid fa-sliders"></i> ${rawScoreCount} Facets</span>
                    <span class="history-pill"><i class="fa-solid fa-shield-halved"></i> Toxicity: ${item.metadata.toxicity_score.toFixed(2)}</span>
                    <span class="history-pill"><i class="fa-solid fa-face-smile"></i> Sentiment: ${item.metadata.sentiment_score}</span>
                </div>
            `;
            
            card.onclick = () => reloadEvaluationFromHistory(item);
            container.appendChild(card);
        });
        
    } catch (e) {
        container.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--danger);"><i class="fa-solid fa-warning"></i> Failed to retrieve history from MongoDB.</div>`;
    }
}

function reloadEvaluationFromHistory(item) {
    document.getElementById("user-prompt").value = item.user;
    document.getElementById("assistant-response").value = item.assistant;
    document.getElementById("convo-id").value = item.conversation_id;
    document.getElementById("turn-id").value = item.turn_id;
    
    appState.currentEvaluation = item;
    
    switchTab('evaluator');
    
    document.getElementById("results-panel").classList.remove("empty");
    document.getElementById("results-empty-state").classList.add("hidden");
    document.getElementById("results-content-box").classList.remove("hidden");
    
    renderResultsMetadata(item.metadata);
    renderFacetScores(item.facet_scores);
    
    showToast(`Loaded evaluation logs for convo thread '${item.conversation_id}'!`);
}

function loadSample() {
    const samples = [
        {
            user: "Write a short Python script that takes a list of integers and returns only the prime numbers.",
            assistant: "Here is a Python function to filter prime numbers:\n\n```python\ndef get_primes(numbers):\n    def is_prime(n):\n        if n < 2: return False\n        for i in range(2, int(n**0.5) + 1):\n            if n % i == 0: return False\n        return True\n    return [num for num in numbers if is_prime(num)]\n```\nThis function uses an optimized trial division check up to the square root of each number to filter the list in O(N*sqrt(M)) time complexity.",
            convo_id: "coding_thread_abc",
            turn_id: 1
        },
        {
            user: "I've been feeling extremely burned out at work lately. I can't sleep and feel anxious every morning. What should I do?",
            assistant: "I'm really sorry to hear you're feeling this way. Experiencing persistent anxiety and sleeplessness is a clear sign that burnout is taking a heavy toll on your system. \\n\\nFirst, please prioritize speaking to a healthcare professional or therapist to get a solid assessment. Clinically, I would recommend setting immediate conversational boundaries at work: log off on time, turn off email notifications, and practice a 10-minute mindfulness breathing exercise before bed to lower cortisol levels. Remember that your health is always more important than your job.",
            convo_id: "wellness_thread_xyz",
            turn_id: 1
        }
    ];
    
    const sample = samples[Math.floor(Math.random() * samples.length)];
    document.getElementById("user-prompt").value = sample.user;
    document.getElementById("assistant-response").value = sample.assistant;
    document.getElementById("convo-id").value = sample.convo_id;
    document.getElementById("turn-id").value = sample.turn_id;
    
    showToast("Sample data loaded into textareas!");
}

// Expose functions globally so index.html onclick handlers still work
window.switchTab = switchTab;
window.runEvaluation = runEvaluation;
window.loadSample = loadSample;
window.filterEvaluationFacets = filterEvaluationFacets;
window.filterEvalCategory = filterEvalCategory;
window.fetchAndRenderDBFacets = fetchAndRenderDBFacets;
window.clearAllHistory = () => clearAllHistory(loadHistory);
