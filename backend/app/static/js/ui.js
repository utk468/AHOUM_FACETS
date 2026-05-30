import { appState } from './state.js';

export function updateStatusIndicator(elementId, isOnline, text) {
    const el = document.getElementById(elementId);
    if (!el) return;
    
    const dot = el.querySelector('.dot');
    const label = el.querySelector('.label');
    
    if (isOnline) {
        el.classList.add('online');
        dot.classList.add('pulse');
        label.innerText = text;
    } else {
        el.classList.remove('online');
        dot.classList.remove('pulse');
        label.innerText = text;
    }
}

export function showToast(message) {
    const toast = document.getElementById("toast");
    const msg = toast.querySelector(".toast-message");
    
    msg.innerText = message;
    toast.classList.remove("hidden");
    
    setTimeout(() => {
        toast.classList.add("hidden");
    }, 4000);
}

export function renderResultsMetadata(meta) {
    document.getElementById("meta-topic").innerText = meta.topic;
    document.getElementById("meta-intent").innerText = meta.intent;
    
    const sentimentEl = document.getElementById("meta-sentiment");
    sentimentEl.innerText = meta.sentiment_score;
    sentimentEl.className = 'meta-value sentiment-badge';
    if (meta.sentiment_score === 'Positive') sentimentEl.classList.add('positive');
    if (meta.sentiment_score === 'Negative') sentimentEl.classList.add('negative');
    
    document.getElementById("meta-toxicity").innerText = meta.toxicity_score.toFixed(3);
    const toxicitySub = document.querySelector("#meta-toxicity + .meta-sub");
    if (meta.toxicity_score > 0.4) {
        toxicitySub.innerText = "Safety Level: RISK DETECTED";
        toxicitySub.style.color = "var(--danger)";
    } else {
        toxicitySub.innerText = "Safety Level: Secure";
        toxicitySub.style.color = "var(--success)";
    }
    
    document.getElementById("meta-readability").innerText = meta.readability_score;
    document.getElementById("meta-time").innerText = `${meta.response_time_estimate}s`;
    document.getElementById("meta-length").innerText = `${meta.turn_length} words`;
    document.getElementById("meta-context").innerText = `${meta.context_window_size} chars`;
}

export function renderFacetScores(scores) {
    const listContainer = document.getElementById("facets-scores-list");
    listContainer.innerHTML = "";
    
    const sortedFacetNames = Object.keys(scores).sort();
    
    sortedFacetNames.forEach(facetName => {
        const item = scores[facetName];
        
        const dbInfo = appState.facetsList.find(f => f.name === facetName) || {
            category: "General",
            description: "No description available."
        };
        
        let dotsHTML = "";
        for (let d = 1; d <= 5; d++) {
            const activeClass = d <= item.score ? "active" : "";
            dotsHTML += `<span class="score-dot ${activeClass}"></span>`;
        }
        
        const card = document.createElement("div");
        card.className = "facet-score-card";
        card.setAttribute("data-name", facetName);
        card.setAttribute("data-category", dbInfo.category);
        card.setAttribute("data-score", item.score);
        card.setAttribute("data-desc", dbInfo.description);
        
        card.innerHTML = `
            <div class="facet-meta">
                <span class="facet-title" title="${facetName}">${facetName}</span>
                <span class="facet-category-tag">${dbInfo.category.split(" ")[0]}</span>
            </div>
            <div class="score-details-row">
                <div class="score-visual">
                    ${dotsHTML}
                </div>
                <div class="score-number">${item.score}<span>/5</span></div>
            </div>
            <div class="facet-confidence-text">
                Confidence: ${(item.confidence * 100).toFixed(0)}%
            </div>
        `;
        
        listContainer.appendChild(card);
    });
    
    filterEvaluationFacets();
}

export function filterEvaluationFacets() {
    const searchEl = document.getElementById("facets-filter-input");
    if (!searchEl) return;
    const searchVal = searchEl.value.toLowerCase();
    const activeCategory = appState.evalFilterCategory;
    
    const cards = document.querySelectorAll("#facets-scores-list .facet-score-card");
    cards.forEach(card => {
        const name = card.getAttribute("data-name").toLowerCase();
        const category = card.getAttribute("data-category");
        
        let matchesSearch = name.includes(searchVal);
        let matchesCategory = true;
        
        if (activeCategory !== 'All') {
            matchesCategory = category.startsWith(activeCategory);
        }
        
        if (matchesSearch && matchesCategory) {
            card.style.display = "flex";
        } else {
            card.style.display = "none";
        }
    });
}

export function filterEvalCategory(category) {
    appState.evalFilterCategory = category;
    
    document.querySelectorAll(".facet-categories-bar .cat-pill").forEach(btn => {
        btn.classList.remove("active");
        if (btn.innerText === category) btn.classList.add("active");
    });
    
    filterEvaluationFacets();
}

export async function fetchAndRenderDBFacets() {
    const search = document.getElementById("db-search-input").value;
    const category = document.getElementById("db-category-select").value;
    
    const tableBody = document.getElementById("db-facets-table");
    tableBody.innerHTML = `<tr><td colspan="3" style="text-align: center; padding: 40px;"><i class="fa-solid fa-spinner fa-spin"></i> Loading facets...</td></tr>`;
    
    try {
        let url = `/api/facets?limit=500`;
        if (search) url += `&search=${encodeURIComponent(search)}`;
        if (category) url += `&category=${encodeURIComponent(category)}`;
        
        const response = await fetch(url);
        if (!response.ok) throw new Error();
        
        const facets = await response.json();
        
        tableBody.innerHTML = "";
        if (facets.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="3" style="text-align: center; padding: 40px; color: var(--text-muted);">No facets matching filter options found.</td></tr>`;
            return;
        }
        
        facets.forEach(f => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${f.name}</td>
                <td><span class="facet-category-tag">${f.category}</span></td>
                <td>${f.description}</td>
            `;
            tableBody.appendChild(row);
        });
    } catch (e) {
        tableBody.innerHTML = `<tr><td colspan="3" style="text-align: center; padding: 40px; color: var(--danger);"><i class="fa-solid fa-warning"></i> Connection failed to MongoDB. Run preprocess.py to seed database.</td></tr>`;
    }
}
