import os
import sys
import json
import faiss
from groq import Groq
from dotenv import load_dotenv

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from ddgs import DDGS

# ==========================================
# 0. LOCAL VS CHAMELEON CONFIG
# ==========================================
# false to try locally with Groq API w/out GPU
# true to run on chameleon with the local model loaded in GPU 
USE_CHAMELEON_GPU = "--gpu" in sys.argv

# Load environment variables
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


if "--vector-dir" in sys.argv:
    try:
        idx = sys.argv.index("--vector-dir")
        SAVE_DIR = sys.argv[idx + 1]
        print(f"VECTOR PATH: Using custom directory -> {SAVE_DIR}")
    except IndexError:
        print("Error: You provided --vector-dir but forgot to type the path!")
        sys.exit(1)
else:
    # AUTOMATIC FALLBACK: This finds the folder so you don't HAVE to type it every time.
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    SAVE_DIR = os.path.join(ROOT_DIR, "crag-vectors")
    print(f"VECTOR PATH: Using default path -> {SAVE_DIR}")


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
    
# ==========================================
# 4. WEB SEARCH TOOL (Dynamic Knowledge Augmentation)
# ==========================================
TRUSTED_DOMAINS = [
    "apple.com", 
    "epa.gov", 
    "macrumors.com", 
    "theverge.com"]
# US Environmental Protection Agency and Apple official site.

def web_search(query: str, num_results: int = 3) -> list[str]:
    """
    Searches the internet using DuckDuckGo, but restricted to only trusted domains.
    """
    print("Starting Web Search...")
    
    # Make filter to only search within trusted domains
    domain_filter = " OR ".join([f"site:{domain}" for domain in TRUSTED_DOMAINS])
    restricted_query = f"{query} ({domain_filter})"
    
    fallback_message = "No information found in web search nor FAISS. Please answer saying that there is not enough information to answer this question."

    results_text = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(restricted_query, max_results=num_results)
            for r in results:
                snippet = r.get("body", "")
                if snippet:
                    results_text.append(snippet)
                    
        if not results_text:
            return [fallback_message]
            
        return results_text
        
    except Exception as e:
        return [fallback_message]