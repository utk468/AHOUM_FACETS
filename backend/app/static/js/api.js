import { appState } from './state.js';
import { updateStatusIndicator, renderResultsMetadata, renderFacetScores, showToast, filterEvaluationFacets } from './ui.js';

// Check connection status of backend components
export async function checkSystemStatus() {
    try {
        const response = await fetch('/api/health');
        if (!response.ok) throw new Error('API down');
        const data = await response.json();
        
        // Update API status
        updateStatusIndicator('status-api', true, 'Connected');
        
        // Update MongoDB status
        updateStatusIndicator('status-db', data.mongodb_connected, data.mongodb_connected ? 'Connected' : 'Offline');
        
        // Update Groq engine status
        updateStatusIndicator('status-groq', data.groq_configured, data.groq_configured ? 'Ready (llama-3)' : 'Mock Fallback');
        
    } catch (e) {
        updateStatusIndicator('status-api', false, 'Offline');
        updateStatusIndicator('status-db', false, 'Offline');
        updateStatusIndicator('status-groq', false, 'Offline');
    }
}

// Load statistics for the top-right indicators
export async function loadDashboardStats() {
    try {
        const response = await fetch('/api/stats');
        if (response.ok) {
            const data = await response.json();
            document.getElementById('stats-total').innerText = data.total_evaluations;
            document.getElementById('stats-conf').innerText = `${data.avg_confidence}%`;
        }
    } catch (e) {
        console.error("Failed to load statistics: ", e);
    }
}

// Fetch seeded facets library from MongoDB on load
export async function fetchDBFacets() {
    try {
        const response = await fetch('/api/facets?limit=1000');
        if (response.ok) {
            appState.facetsList = await response.json();
            console.log(`Loaded ${appState.facetsList.length} facets from library.`);
        }
    } catch (e) {
        console.error("Could not load facets database: ", e);
    }
}

// Trigger Pipeline Evaluation
export async function runEvaluation() {
    const userPrompt = document.getElementById("user-prompt").value.trim();
    const assistantResponse = document.getElementById("assistant-response").value.trim();
    const convoId = document.getElementById("convo-id").value;
    const turnId = parseInt(document.getElementById("turn-id").value) || 1;
    
    if (!userPrompt || !assistantResponse) {
        showToast("Please enter both a User query and Assistant response!");
        return;
    }
    
    const btn = document.getElementById("btn-evaluate");
    btn.classList.add("loading");
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing 300+ Facets...';
    
    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user: userPrompt,
                assistant: assistantResponse,
                conversation_id: convoId || null,
                turn_id: turnId,
                history: []
            })
        });
        
        if (!response.ok) throw new Error("API call failed");
        
        const data = await response.json();
        appState.currentEvaluation = data;
        
        // Show panel and hide empty state
        document.getElementById("results-panel").classList.remove("empty");
        document.getElementById("results-empty-state").classList.add("hidden");
        document.getElementById("results-content-box").classList.remove("hidden");
        
        // Render metrics & facets
        renderResultsMetadata(data.metadata);
        renderFacetScores(data.facet_scores);
        
        // Update stats
        loadDashboardStats();
        
        showToast("Turn successfully evaluated across all facets!");
        
    } catch (e) {
        showToast("Evaluation failed. Make sure server is running.");
        console.error(e);
    } finally {
        btn.classList.remove("loading");
        btn.innerHTML = '<i class="fa-solid fa-rocket"></i> Run High-Fidelity Evaluation';
    }
}

// Clear history
export async function clearAllHistory(loadHistoryCallback) {
    if (!confirm("Are you absolutely sure you want to delete all historical evaluation logs? This is irreversible!")) {
        return;
    }
    
    try {
        const response = await fetch('/api/clear', { method: 'POST' });
        if (response.ok) {
            showToast("Historical logs cleared from database!");
            loadHistoryCallback();
            loadDashboardStats();
            
            // Reset pane view
            document.getElementById("results-panel").classList.add("empty");
            document.getElementById("results-empty-state").classList.remove("hidden");
            document.getElementById("results-content-box").classList.add("hidden");
        }
    } catch (e) {
        showToast("Clear command failed.");
    }
}
