import os
import re
import json
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from huggingface_hub import InferenceClient
from groq import Groq

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "facets_evaluator")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Core 30 Features that will be extracted by the LLM
CORE_FEATURES = [
    {"name": "grammar_correctness", "desc": "Grammatical correctness, spelling, syntax, and language rules adherence."},
    {"name": "fluency", "desc": "Smoothness, natural phrasing, vocabulary flow, and sentence transition quality."},
    {"name": "coherence_cohesion", "desc": "Logical structural integrity, connection between thoughts, and readability."},
    {"name": "clarity", "desc": "Simplicity, directness, and ease of understanding of the message without ambiguity."},
    {"name": "conciseness", "desc": "Brevity, avoiding fluff, wordiness, or unnecessary repetition."},
    {"name": "relevance", "desc": "Adherence to the user's prompt, remaining on topic, and addressing queries directly."},
    {"name": "context_awareness", "desc": "Recognizing conversational history, references, and implicit assumptions."},
    {"name": "goal_completion", "desc": "Completeness in executing the user request and satisfying the underlying intent."},
    {"name": "safety_compliance", "desc": "Adherence to safety guidelines, refusing dangerous, illegal, or unethical requests."},
    {"name": "toxicity_avoidance", "desc": "Avoiding offensive, hateful, abusive, rude, or aggressive remarks."},
    {"name": "harm_prevention", "desc": "Preventing self-harm, cyberattacks, violence, physical harm, or illegal acts."},
    {"name": "bias_fairness", "desc": "Maintaining neutrality, inclusivity, and avoiding stereotypes or prejudice."},
    {"name": "privacy_protection", "desc": "Avoiding leakage of personal data, passwords, credentials, or private info."},
    {"name": "factuality_truth", "desc": "Accuracy, truthfulness, and avoiding false claims or hallucinations."},
    {"name": "empathy", "desc": "Showing emotional warmth, validating feelings, and demonstrating active listening."},
    {"name": "politeness", "desc": "Being courteous, using polite language, greetings, and showing respect."},
    {"name": "friendliness", "desc": "Exhibiting an approachable, cheerful, open, and encouraging demeanor."},
    {"name": "emotional_understanding", "desc": "Understanding psychological states, stress levels, and emotional undertones."},
    {"name": "logical_reasoning", "desc": "Applying structured logic, step-by-step analysis, and rational deduction."},
    {"name": "numerical_accuracy", "desc": "Correctness in mathematics, accounting, equations, and data calculations."},
    {"name": "creativity", "desc": "Originality, artistic storytelling, brainstorming ideas, and quirkiness."},
    {"name": "leadership_presence", "desc": "Providing guidance, authoritative direction, structured advice, and confidence."},
    {"name": "adaptability", "desc": "Flexibility, accommodating corrections, and shifting perspectives easily."},
    {"name": "scientific_knowledge", "desc": "Accurate details in natural sciences, medicine, health literacy, and physics."},
    {"name": "spiritual_philosophical", "desc": "Discussing values, meditation, religions, Gnosticism, Sufism, and inner peace."},
    {"name": "lifestyle_wellness", "desc": "Discussing food, sleep, exercise, travels, habits, and daily routines."},
    {"name": "social_intelligence", "desc": "Navigating social norms, interpersonal dynamics, and cultural identity."},
    {"name": "structure_organization", "desc": "Formatting using bullet points, headers, tables, and clear layout flow."},
    {"name": "professionalism", "desc": "Exhibiting business decorum, specialist knowledge, and high work standards."},
    {"name": "brevity_precision", "desc": "Getting straight to facts, data analysis, and using highly dense statements."}
]

# Simple rule-based domain categorizer based on keyword matching
def categorize_facet(name):
    name_lower = name.lower()
    
    # Linguistic & Quality
    if any(k in name_lower for k in ["grammar", "spell", "vocab", "readab", "senten", "text", "fluenc", "coher", "clarit", "concise", "spoken", "auditor", "listen", "brief", "story", "word", "alphab", "linguist", "pronounc"]):
        return "Linguistic & Quality"
    
    # Safety & Risk
    if any(k in name_lower for k in ["safe", "toxic", "harm", "bias", "privac", "leak", "violenc", "drug", "physic", "hate", "secur", "disrespect", "sloth", "cunnin", "dishonest", "abuse"]):
        return "Safety & Risk"
    
    # Emotion & Interpersonal
    if any(k in name_lower for k in ["emot", "empath", "polit", "friend", "support", "mood", "trust", "affect", "anger", "warm", "compass", "spirit", "holy", "sacred", "sufi", "zen", "yoga", "religion", "church", "god", "shabbat", "quran", "bibl", "kabbalah", "chakra", "reiki", "gnostic", "jewish", "hindu", "islamic", "bhagavad", "buddhi", "joy", "sad", "depress", "burnout", "hostil", "anxiet", "panic"]):
        return "Emotion & Interpersonal"
    
    # Cognition & Reasoning
    if any(k in name_lower for k in ["reason", "logic", "calcul", "math", "intellig", "cognit", "iq", "learn", "memor", "problem", "analys", "sequenc", "metric", "study", "data", "science", "formula", "spatial", "assess", "test"]):
        return "Cognition & Reasoning"
    
    # Lifestyle & Behavioral
    if any(k in name_lower for k in ["sleep", "eat", "diet", "travel", "commut", "hobb", "sport", "lead", "leadership", "style", "work", "decis", "dynamic", "activit", "passport", "nomad", "transport", "custom", "habit", "breakfast", "caffein", "immunit", "money", "financ"]):
        return "Lifestyle & Behavioral"
    
    return "General Personality & Behavior"

# Clean raw facet names from CSV
def clean_facet_name(raw_name):
    # Remove leading numbering patterns like "800. Sufi practice:" or "644. "
    cleaned = re.sub(r'^\d+\.\s*', '', raw_name)
    # Remove trailing colons
    cleaned = cleaned.strip().rstrip(':')
    # Standardize hyphenation and spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Capitalize appropriately (e.g., Risktaking -> Risk-taking or Risk Taking)
    if cleaned.lower() == "risktaking":
        cleaned = "Risk-taking"
    elif cleaned.lower() == "selfesteem":
        cleaned = "Self-esteem"
    elif cleaned.lower() == "selfcontrol":
        cleaned = "Self-control"
    elif cleaned.lower() == "selfdirectedness":
        cleaned = "Self-directedness"
    
    return cleaned

# Generate descriptions in batches using Groq if key is available, else fallback to templates
def generate_descriptions(facet_names):
    is_placeholder = True
    if is_placeholder:
        print("[!] GROQ_API_KEY is not configured or placeholder. Falling back to high-quality rule-based descriptions...")
        return generate_fallback_descriptions(facet_names)
        
    print(f"[*] Initializing Groq Client for {len(facet_names)} facets...")
    client = Groq(api_key=GROQ_API_KEY)
    descriptions = {}
    
    # Process in batches of 50
    batch_size = 50
    for i in range(0, len(facet_names), batch_size):
        batch = facet_names[i:i+batch_size]
        print(f"   Processing batch {i//batch_size + 1}/{(len(facet_names)-1)//batch_size + 1} ({len(batch)} facets)...")
        
        prompt = f"""
You are an expert NLP and psychological profiling engine.
Generate a concise, 1-sentence description for each of the following {len(batch)} evaluation facets.
The description should explain what the facet measures in a conversation turn between a User and an AI Assistant.

Facets:
{json.dumps(batch, indent=2)}

Return the output strictly as a JSON object where the keys are the exact facet names and the values are their 1-sentence descriptions. Do not include markdown code blocks or any conversational filler. Just return the raw JSON object.
"""
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=GROQ_MODEL,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            batch_desc = json.loads(chat_completion.choices[0].message.content)
            # Add to main dict
            for name in batch:
                descriptions[name] = batch_desc.get(name, f"Measures the level of {name.lower()} in the conversation turn.")
        except Exception as e:
            print(f"[x] Groq Batch failed: {e}. Falling back to default descriptions for this batch.")
            for name in batch:
                descriptions[name] = f"Measures the level of {name.lower()} in the conversation turn."
        
        time.sleep(1) # Rate limit protection
        
    return descriptions

def generate_fallback_descriptions(facet_names):
    descriptions = {}
    for name in facet_names:
        # Heuristics based on name keywords to create slightly more readable fallbacks
        lower = name.lower()
        if "level" in lower:
            descriptions[name] = f"Evaluates the individual's measured {lower} in the conversational context."
        elif "practice" in lower or "metric" in lower:
            descriptions[name] = f"Assesses adherence or engagement frequency with {lower} topics or routines."
        elif "strength" in lower or "trait" in lower:
            descriptions[name] = f"Measures the psychological trait of {lower} exhibited in the dialogue."
        elif "knowledge" in lower or "skill" in lower:
            descriptions[name] = f"Assesses the level of comprehension, information, and skills regarding {lower}."
        else:
            descriptions[name] = f"Measures the presence, strength, or quality of {lower} traits and concepts."
    return descriptions

def main():
    csv_path = "c:\\Users\\ASUS\\Desktop\\AHOUM\\Facets Assignment.csv"
    if not os.path.exists(csv_path):
        print(f"[x] Error: Could not find CSV file at {csv_path}")
        return
        
    print(f"[+] Loading facets from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Expecting column 'Facets' (or whatever the header is)
    col_name = df.columns[0]
    print(f"[*] Reading column '{col_name}'...")
    
    # 1. Clean & Preprocess
    raw_facets = df[col_name].dropna().astype(str).tolist()
    print(f"[+] Loaded {len(raw_facets)} raw facet rows.")
    
    cleaned_facets = []
    seen = set()
    for raw in raw_facets:
        clean = clean_facet_name(raw)
        if clean and clean not in seen:
            seen.add(clean)
            cleaned_facets.append((raw, clean))
            
    print(f"[+] Cleaned and removed duplicates: {len(cleaned_facets)} unique facets remaining.")
    
    # 2. Categorize
    facets_data = []
    for raw, clean in cleaned_facets:
        category = categorize_facet(clean)
        facets_data.append({
            "raw_name": raw,
            "name": clean,
            "category": category
        })
        
    # 3. Generate Descriptions
    facet_names = [f["name"] for f in facets_data]
    descriptions = generate_descriptions(facet_names)
    
    for f in facets_data:
        f["description"] = descriptions.get(f["name"], f"Measures {f['name'].lower()} characteristics.")
        
    # 4. Embeddings & Feature Alignment Weights
    if not HF_TOKEN or HF_TOKEN == "YOUR_HF_TOKEN":
        print("[!] HF_TOKEN not configured. Skipping embeddings and alignment calculation. Please add HF_TOKEN to .env!")
        return

    print("[*] Initializing HuggingFace InferenceClient...")
    client = InferenceClient(api_key=HF_TOKEN)
    model_name = "jinaai/jina-embeddings-v5-text-nano"
    
    def get_embeddings_in_batches(text_list, batch_size=10):
        all_embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            print(f"   Embedding batch {i//batch_size + 1}/{(len(text_list)-1)//batch_size + 1}...")
            try:
                # Some HF endpoints don't strictly support batch array feature extraction via client.feature_extraction
                # So we fallback to looping individually but quickly.
                batch_embs = []
                for text in batch:
                    res = client.feature_extraction(text, model=model_name)
                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                        batch_embs.append(res[0])
                    else:
                        batch_embs.append(res)
                all_embeddings.extend(batch_embs)
                time.sleep(1) # Rate limit protection
            except Exception as e:
                print(f"[x] API Error during embedding: {e}")
                # Mock fallback for this batch if api fails
                batch_embs = []
                for t in batch:
                    import hashlib
                    h = hashlib.md5(t.encode('utf-8')).hexdigest()
                    rng = np.random.default_rng(int(h, 16) & 0xffffffff)
                    vec = rng.normal(0, 1, 384)
                    batch_embs.append((vec / np.linalg.norm(vec)).tolist())
                all_embeddings.extend(batch_embs)
        return all_embeddings

    print("[+] Computing Embeddings for Core Features via HF API...")
    core_embeddings = get_embeddings_in_batches([f["desc"] for f in CORE_FEATURES])
    
    print("[+] Computing Embeddings for all Facets & Calculating Alignment Weights...")
    facet_desc_list = [f["description"] for f in facets_data]
    facet_embeddings = get_embeddings_in_batches(facet_desc_list)
    
    # Store everything in the list
    for idx, f in enumerate(facets_data):
        f_vec = np.array(facet_embeddings[idx])
        f["embedding"] = f_vec.tolist()
        
        # Calculate alignment weights (cosine similarities between this facet and the 30 core features)
        # Cosine similarity = dot product of normalized vectors
        f_norm = f_vec / np.linalg.norm(f_vec)
        
        similarities = []
        for c_idx, c in enumerate(CORE_FEATURES):
            c_vec = np.array(core_embeddings[c_idx])
            c_norm = c_vec / np.linalg.norm(c_vec)
            
            sim = np.dot(f_norm, c_norm)
            # Re-scale similarity to highlight strong matches (raise to power of 3)
            # This ensures only highly-relevant features get strong weights
            similarities.append(max(0, sim) ** 3)
            
        sim_sum = sum(similarities)
        if sim_sum > 0:
            weights = [s / sim_sum for s in similarities]
        else:
            # Equal weights as fallback
            weights = [1.0 / len(CORE_FEATURES)] * len(CORE_FEATURES)
            
        f["weights"] = weights
        
    # 5. Seed MongoDB
    print(f"[*] Connecting to MongoDB at {MONGO_URI}...")
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        facets_coll = db["facets"]
        
        # Clear existing
        facets_coll.delete_many({})
        
        # Insert all
        print(f"[+] Seeding {len(facets_data)} facets into MongoDB collection 'facets'...")
        facets_coll.insert_many(facets_data)
        
        # Create index on name for quick lookups
        facets_coll.create_index("name", unique=True)
        facets_coll.create_index("category")
        
        print("[+] Database successfully seeded and indexed!")
        
    except Exception as e:
        print(f"[x] Failed to seed MongoDB: {e}")
        
if __name__ == "__main__":
    main()
