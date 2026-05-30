// Global App State
export const appState = {
    totalEvaluations: 0,
    avgConfidence: 0,
    facetsList: [],          // Seeded facet list from DB
    currentEvaluation: null, // Full response object from last API evaluation
    activeTab: 'evaluator',
    evalFilterText: '',
    evalFilterCategory: 'All'
};
