# AETHER: Scalable Facets Evaluation System

AETHER is a high-performance, modular evaluation engine designed to assess conversation turns across **300+ dimensions (scalable to 5,000+ facets)** in microseconds. Built with **FastAPI, MongoDB, HTML/CSS/JS, and Sentence Embeddings**, it circumvents the slow and expensive rate limits of one-shot LLM prompting by utilizing an intelligent **Feature Extraction + Matrix Scoring** architecture.

---

## 🚀 Architectural Blueprint

AETHER avoids the architectural bottleneck of executing separate LLM queries per evaluation facet (which breaks when scaling to 5,000+ facets) by decoupling text understanding from facet mapping:

1. **Local Text Embedding (`all-MiniLM-L6-v2`)**: Generates a fast 384-dimensional semantic representation of the conversation turn.
2. **LLM Feature Extraction (Llama-3-8B via Groq)**: A single parallelized LLM call extracts a dense vector of **30 core attributes** (covering linguistic quality, safety, and emotional indicators) and conversation turn metadata (sentiment, toxicity, readability, topic, intent).
3. **Similarity Weight Mapping Matrix ($W$)**: During database seeding, each facet's description is embedded and matched against the 30 core features. This pre-calculates an alignment weight vector for every facet, stored in MongoDB.
4. **Vector Scoring Layer ($W \times F$)**: The final score is computed instantly by taking the dot-product of the LLM's extracted feature vector ($F$) and the facet's similarity weights matrix ($W$), blended with direct semantic relevance. This enables scaling to 5,000+ facets in microseconds with **zero extra LLM calls**!
5. **Deterministic Mock Fallback**: If the `GROQ_API_KEY` environment variable is not provided, the engine gracefully degrades to a deterministic, high-variance mock fallback system. This ensures the system never crashes and always generates a realistic spread of 1-5 scores for offline testing or demonstrations.

---

## 🛠️ Tech Stack & Key Files

- **Backend**: FastAPI, PyMongo, SentenceTransformers (local), Groq SDK (LLM integration)
- **Frontend**: Clean Vanilla HTML5, CSS3 (Premium dark black theme with vibrant orange accents), Vanilla JS
- **Database**: MongoDB (Facets library & Conversation history store)
- **Key Modules**:
  - `preprocess.py`: Data cleaning, category classification, semantic weight generation, and MongoDB seeding.
  - `backend/app/services/extractor.py`: Structural metadata and core feature extraction.
  - `backend/app/services/scorer.py`: Fast matrix dot-product scoring and embedding blending.
  - `backend/app/services/pipeline.py`: Orchestrator of the full evaluation cycle.

---

## 💻 Quickstart Guide (Local Setup)

### 1. Prerequisites
- **Python**: v3.11+
- **MongoDB**: Installed and running locally (`mongodb://localhost:27017/`)
- **Groq API Key**: Create a free account and get a key at [Groq Console](https://console.groq.com/)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Configure your credentials in the `.env` file (copied from `.env.template`):
```ini
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=llama3-8b-8192
MONGO_URI=mongodb://localhost:27017/
MONGO_DB=facets_evaluator
PORT=8000
```

### 4. Clean CSV & Seed MongoDB
The `preprocess.py` script automatically parses `Facets Assignment.csv`, cleans the names (removing numbers like `800.` and trailing colons), automatically categorizes them into 5 domains, generates descriptions, computes embeddings, and populates MongoDB:
```bash
python preprocess.py
```

### 5. Launch the Application
Start the FastAPI server:
```bash
python app.py
```
Open your browser and navigate to: **[http://localhost:8000](http://localhost:8000)**

---

## 🐳 Dockerized Baseline (Brownie Point 2)

AETHER is fully dockerized for instant single-command deployment.

### 1. Build the evaluator
```bash
docker build -t evaluator .
```

### 2. Run the evaluator
Ensure a MongoDB instance is available, or use the Docker Compose orchestration below:
```bash
docker run -p 8000:8000 -e GROQ_API_KEY=your_key evaluator
```

### 3. Deploy with Docker Compose (Recommended)
This spins up both the FastAPI application and a configured MongoDB instance in isolated containers:
```bash
# Provide key inline or ensure it is set in your host environment
GROQ_API_KEY=your_key docker-compose up --build
```
Navigate to **[http://localhost:8000](http://localhost:8000)** to explore!

---

## 📡 API Documentation

AETHER exposes clean, self-documenting JSON endpoints (accessible interactively at `/docs`):

- **`POST /api/evaluate`**:
  Submit a turn for evaluation.
  - Request:
    ```json
    {
      "user": "I am feeling burnt out at work.",
      "assistant": "I am sorry to hear that. You should take a break.",
      "conversation_id": "optional-uuid",
      "turn_id": 1
    }
    ```
  - Response: Evaluates all 300+ facets and returns scores, confidence levels, and extra column metadata (sentiment, toxicity, readability, topic, intent, complexity).
- **`GET /api/facets`**: Fetch the list of assessment facets. Supports `search` (text matching) and `category` query filters.
- **`GET /api/history`**: Get past evaluated turns stored in MongoDB.
- **`GET /api/stats`**: Get overall scoring averages and common topic statistics.
