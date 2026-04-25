import os
import json
import faiss
from groq import Groq
from dotenv import load_dotenv

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 0. LOCAL VS CHAMELEON CONFIG
# ==========================================
# false to try locally with Groq API w/out GPU
# true to run on chameleon with the local model loaded in GPU 
USE_CHAMELEON_GPU = False 

# Load environment variables
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#constants
SAVE_DIR    = "../baseline/crag-vectors"
INDEX_PATH  = os.path.join(SAVE_DIR, "corpus_index.faiss")
IDMAP_PATH  = os.path.join(SAVE_DIR, "id_map.json")

ENCODER_MODEL = "all-MiniLM-L6-v2"
GENERATOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct" 
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

# In chameleon we load the generator model locally on GPU for faster inference. In local testing, we will use the Groq API for generation.  
if USE_CHAMELEON_GPU:
    print(f"Loading {GENERATOR_MODEL} on GPU...")
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    gen_model     = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda:0", 
    )
    gen_model.eval()


# ==========================================
# 2. RETRIEVAL TOOL (From Baseline)
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
            "chunk_id": entry["chunk_id"], 
            "score": float(dist),
        })
    return results


# ==========================================
# 3. GENERATION TOOL (From Baseline)
# ==========================================
# Difference from Baseline: LangGraph gives us a list of strings, so we changed
# this function to accept strings instead of dictionaries to match it perfectly. 

SYSTEM_PROMPT = "You are a helpful assistant. Answer the question based on the provided context."

def generate(query: str, context_list: list[str]) -> str:
    """
    Generates an answer using either Local HF (Chameleon) or Groq API (Local).
    """
    
    context_str = "\n\n".join(context_list)
    
    if USE_CHAMELEON_GPU:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
        ]
        inputs = gen_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(gen_model.device)

        with torch.no_grad():
            output_ids = gen_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=gen_tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return gen_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    else:
        # Groq logic for local testing
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