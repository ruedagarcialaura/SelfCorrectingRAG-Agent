import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Constants from your teammate's baseline
SAVE_DIR    = "../baseline/crag-vectors"
INDEX_PATH  = f"{SAVE_DIR}/corpus_index.faiss"
IDMAP_PATH  = f"{SAVE_DIR}/id_map.json"
ENCODER_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.1-8b-instant"

# ==========================================
# 1. INITIALIZE MODELS AND FAISS INDEX
# ==========================================
print("Loading Encoder and FAISS index in tools.py...")
try:
    encoder = SentenceTransformer(ENCODER_MODEL)
    index = faiss.read_index(INDEX_PATH)
    id_map = json.load(open(IDMAP_PATH, encoding="utf-8"))
    print(f"FAISS loaded successfully: {index.ntotal} vectors.")
except Exception as e:
    print(f"Warning: Could not load FAISS index. Make sure '{SAVE_DIR}' folder exists. Error: {e}")


# ==========================================
# 2. RETRIEVAL TOOL (From Baseline Cel 5)
# ==========================================
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Encodes the query and performs FAISS L2 search.
    Returns the top-k most similar sub-chunks.
    """
    q_vec = encoder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        entry = id_map[str(idx)]
        results.append({
            "text": entry["text"],
            "score": float(dist),
        })
    return results


# ==========================================
# 3. GENERATION TOOL (From Baseline Cel 6)
# ==========================================
def generate(query: str, context_list: list[str]) -> str:
    """
    Generates an answer using Groq / Llama 3.1 based on the provided context.
    Notice we pass a list of strings here to match our LangGraph State.
    """
    SYSTEM_PROMPT = "You are a helpful assistant. Answer the question based on the provided context."
    context_str = "\n\n".join(context_list)
    user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"
    
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,     
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()