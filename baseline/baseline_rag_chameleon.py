# ==============================================================================
# # baseline_rag_chameleon.ipynb — Baseline RAG Pipeline (Chameleon GPU)
# **Project:** Self-Corrective RAG (CRAG) — ITMD 524, Spring 2026  
# **Purpose:** Run the full baseline RAG evaluation on Chameleon with a local Llama 3.1-8B generator.  
# **Environment:** Chameleon node with GPU, conda `crag` env, artifacts at `~/crag-vectors/`.
# 
# ## Before running this notebook
# 
# 1. **Environment setup** — Run `setup_env.sh` on the Chameleon node (installs conda + packages).
# 2. **FAISS artifacts** — Transfer from Google Drive via rclone:
#    ```
#    rclone copy gdrive:crag-vectors/ ~/crag-vectors/
#    ```
#    You should see `corpus_index.faiss` (~1.7 MB), `id_map.json` (~870 KB), `config.json` (~1 KB).
# 3. **`.env` file** — Create `~/.env` with your Groq API key (for RAGAS judge only):
#    ```
#    echo "GROQ_API_KEY=your_key_here" > ~/.env
#    ```
# 4. **Pre-download Llama weights** (10–15 min):
#    ```python
#    python3 -c "
#    from transformers import AutoTokenizer, AutoModelForCausalLM
#    AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
#    AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
#    "
#    ```
# 5. **Run all cells top-to-bottom.**
# ==============================================================================

# ── Cell 1: Constants + Local Paths (Chameleon) ────────────────────────────────
# Change 1: No Google Drive mount — artifacts are at ~/crag-vectors/
import os

ARTIFACTS_DIR = os.path.expanduser("~/crag-vectors")
INDEX_PATH    = f"{ARTIFACTS_DIR}/corpus_index.faiss"
IDMAP_PATH    = f"{ARTIFACTS_DIR}/id_map.json"
CONFIG_PATH   = f"{ARTIFACTS_DIR}/config.json"
RESULTS_PATH  = os.path.expanduser("~/baseline_results.csv")

ENCODER_MODEL   = "all-MiniLM-L6-v2"
GENERATOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TOP_K           = 5
EVAL_SAMPLES    = 800
HF_DATASET_ID   = "AdamLucek/apple-environmental-report-QA-retrieval"

print(f"Artifacts dir   : {ARTIFACTS_DIR}")
print(f"Encoder model   : {ENCODER_MODEL}")
print(f"Generator model : {GENERATOR_MODEL}")
print(f"TOP_K           : {TOP_K}")
print(f"Eval samples    : {EVAL_SAMPLES}")

# ==============================================================================
# ## Step 1 — Verify Environment
# 
# All dependencies were installed via `setup_env.sh` (terminal). This cell verifies they are available.
# ==============================================================================

# ── Cell 2: Verify Installed Packages (Chameleon) ─────────────────────────────
# Change 2: No pip install — just verify what setup_env.sh installed
import faiss, sentence_transformers, transformers, torch, datasets, ragas
print(f"faiss                 {faiss.__version__}")
print(f"sentence-transformers {sentence_transformers.__version__}")
print(f"transformers          {transformers.__version__}")
print(f"torch                 {torch.__version__}")
print(f"GPU available         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU                   {torch.cuda.get_device_name(0)}")

# ==============================================================================
# ## Step 2 — Load Groq API Key
# 
# Loaded from `~/.env`. The key is only needed for RAGAS scoring (Groq judge), not for generation.
# Loading it early catches missing keys before the long evaluation run.
# ==============================================================================

# ── Cell 3: Load API Key from ~/.env (Chameleon) ──────────────────────────────
# Change 3: Path changed from /content/.env to ~/.env
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "GROQ_API_KEY not found in ~/.env"
print(f"Groq API key loaded (starts with: {GROQ_API_KEY[:8]}...)")

# ==============================================================================
# ## Step 3 — Load FAISS Index and ID Map
# ==============================================================================

# ── Cell 4: Load FAISS Index + id_map from ~/crag-vectors/ ───────────────────
import faiss
import json

index  = faiss.read_index(INDEX_PATH)
id_map = json.load(open(IDMAP_PATH, encoding="utf-8"))

assert index.ntotal == len(id_map), (
    f"Mismatch: index has {index.ntotal} vectors but id_map has {len(id_map)} entries."
)

print(f"FAISS index loaded: {index.ntotal} vectors, dim={index.d}")
print(f"id_map loaded     : {len(id_map)} entries")

# Print build config for traceability
config = json.load(open(CONFIG_PATH))
print(f"\nCorpus config:")
print(f"  encoder_model    : {config['encoder_model']}")
print(f"  chunk_size_tokens: {config.get('chunk_size_tokens', 'N/A')}")
print(f"  n_source_chunks  : {config['n_source_chunks']}")
print(f"  n_sub_chunks     : {config['n_sub_chunks']}")
print(f"  build_date       : {config['build_date']}")

# ==============================================================================
# ## Step 4 — Retriever
# 
# Encodes the query with `all-MiniLM-L6-v2` and performs exact L2 search against the 1158-vector FAISS index.  
# Returns the top-k most similar sub-chunks with their text and metadata.
# ==============================================================================

# ── Cell 5: Encoder + retrieve() ───────────────────────────────────
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer(ENCODER_MODEL)

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return top-k chunks for a query via exact FAISS L2 search."""
    q_vec = encoder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        entry = id_map[str(idx)]
        results.append({
            "text":     entry["text"],
            "chunk_id": entry["chunk_id"],
            "score":    float(dist),   # L2 distance (lower = more similar)
        })
    return results

# ── Smoke test ────────────────────────────────────────────────────────────────
sample = retrieve("Apple renewable energy percentage", top_k=3)
print(f"retrieve() — {len(sample)} results")
for i, r in enumerate(sample, 1):
    print(f"  [{i}] {r['chunk_id']} | score={r['score']:.4f} | '{r['text'][:80]}'")


# ==============================================================================
# ## Step 5 — Generator (Local Llama 3.1-8B)
# 
# Uses a **local** `meta-llama/Llama-3.1-8B-Instruct` model loaded on GPU via HuggingFace Transformers.  
# 
# **Key details:**
# - `apply_chat_template` formats the prompt correctly for Llama 3 instruct format (handles `<|begin_of_text|>` tokens automatically).
# - `do_sample=False` with `temperature=1.0` gives deterministic greedy decoding — same behavior as `temperature=0` in Groq.
# - **System prompt is minimal by design** — no "I don't know" escape hatch. The baseline must attempt
# to answer even with irrelevant chunks so that hallucinations surface in the Faithfulness score
# during evaluation. This is the failure mode the CRAG Router is designed to prevent.
# ==============================================================================

# ── Cell 6: Local Llama Generator (Chameleon GPU) ────────────────────────────
# Change 4: Replaces Groq API client with local HuggingFace pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"Loading {GENERATOR_MODEL} on GPU...")

gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
gen_model     = AutoModelForCausalLM.from_pretrained(
    GENERATOR_MODEL,
    torch_dtype=torch.float16,
    device_map="cuda:0",
)
gen_model.eval()

SYSTEM_PROMPT = "You are a helpful assistant. Answer the question based on the provided context."

def generate(query: str, context_chunks: list[dict]) -> str:
    """Generate an answer from retrieved chunks using local Llama 3.1-8B."""
    context  = "\n\n".join(c["text"] for c in context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
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

# Smoke test
answer = generate("What is Apple's renewable energy percentage?", sample)
print("generate() — ok")
print(f"\nAnswer: {answer}")

# ==============================================================================
# ## Step 6 — Full Baseline Pipeline
# 
# Chains retrieve + generate into a single function. Two smoke tests verify the end-to-end pipeline.
# ==============================================================================

# ── Cell 7: baseline_rag() — Full Pipeline ────────────────────────────────────
def baseline_rag(query: str, top_k: int = TOP_K) -> dict:
    """Full baseline pipeline: retrieve → generate (no validation)."""
    chunks = retrieve(query, top_k=top_k)
    answer = generate(query, chunks)
    return {
        "answer":   answer,
        "contexts": [c["text"] for c in chunks],  # list[str] required by RAGAS
        "chunks":   chunks,                         # full metadata for debugging
    }

# ── End-to-end smoke tests ────────────────────────────────────────────────────
queries = [
    "How does Apple track its carbon emissions?",
    "What percentage of Apple's suppliers have committed to clean energy?",
]

for q in queries:
    result = baseline_rag(q)
    print(f"Q: {q}")
    print(f"A: {result['answer'][:300]}")
    print(f"   → {len(result['contexts'])} chunks retrieved")
    print()

print("baseline_rag() — pipeline verified")

# ==============================================================================
# ## Step 7 — 800-Sample Batch Evaluation
# 
# Runs `baseline_rag()` on 800 deduplicated questions from the validation split.  
# Checkpoints every 50 rows to protect against node interruption.  
# Prints elapsed/remaining time estimates for monitoring.
# ==============================================================================

# ── Cell 8: 800-Sample Batch Evaluation ──────────────────────────────────────
# Change 5: New cell — full evaluation with incremental saving (Batched for Speed)
import pandas as pd
import time
import torch
from datasets import load_dataset

ds  = load_dataset(HF_DATASET_ID)
val = pd.DataFrame(ds["validation"]).drop_duplicates(subset="question")
val = val.sample(n=EVAL_SAMPLES, random_state=42).reset_index(drop=True)

print(f"Running batched evaluation on {EVAL_SAMPLES} samples...")
print(f"Results will be saved to: {RESULTS_PATH}\n")

records = []
start   = time.time()
BATCH_SIZE = 4  # Reduced batch size to prevent GPU OOM or slow CPU offloading

# Configure tokenizer for batching (left-padding required for decoder-only models)
if getattr(gen_tokenizer, 'pad_token', None) is None:
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
gen_tokenizer.padding_side = 'left'

for i in range(0, len(val), BATCH_SIZE):
    batch = val.iloc[i:i+BATCH_SIZE]
    
    # 1. Sequential retrieval (FAISS is fast enough on CPU)
    batch_contexts = []
    batch_chunks = []
    for q in batch["question"]:
        chunks = retrieve(q, top_k=TOP_K)
        batch_chunks.append(chunks)
        batch_contexts.append([c["text"] for c in chunks])
        
    # 2. Batch prompt construction
    batch_messages = []
    for q, chunks in zip(batch["question"], batch_chunks):
        context = "\n\n".join([c["text"] for c in chunks])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {q}"},
        ]
        batch_messages.append(messages)
        
    # 3. Batch tokenize and generate on GPU
    texts = [gen_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in batch_messages]
    inputs = gen_tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False).to(gen_model.device)
    
    with torch.no_grad():
        output_ids = gen_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=gen_tokenizer.pad_token_id,
        )
        
    # 4. Decode new tokens only
    new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
    answers = gen_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    
    # 5. Save results
    for j, (idx, row) in enumerate(batch.iterrows()):
        records.append({
            "question":     row["question"],
            "ground_truth": row["chunk"],
            "answer":       answers[j].strip(),
            "contexts":     batch_contexts[j],
        })
        
    # 6. Logging and Checkpointing
    processed = i + len(batch)
    elapsed = time.time() - start
    avg = elapsed / processed
    remaining = avg * (EVAL_SAMPLES - processed)
    print(f"[{processed:03d}/{EVAL_SAMPLES}] Batched elapsed: {elapsed/60:.1f} min | remaining: {remaining/60:.1f} min")

    if processed % 48 == 0 or processed == len(val):
        pd.DataFrame(records).to_csv(RESULTS_PATH, index=False)
        print(f"         checkpoint saved ({processed} rows)")

pd.DataFrame(records).to_csv(RESULTS_PATH, index=False)
print(f"\nBatch complete. {len(records)} samples saved to {RESULTS_PATH}\n")


# ==============================================================================
# ## Step 8 — RAGAS Evaluation
# 
# Scores the 800 baseline answers using RAGAS metrics (Faithfulness, Answer Relevancy).  
# 
# **Judge configuration:**
# - **Generator (already loaded):** Local Llama 3.1-8B on GPU.
# - **RAGAS judge:** Groq API (`llama-3.1-8b-instant`) — same model, much simpler to wrap as a RAGAS judge than a local HuggingFace pipeline.
# 
# **Why Groq for the judge?** Wrapping a local HuggingFace pipeline as a RAGAS-compatible LLM requires complex formatting and is unreliable. Groq uses the same Llama 3.1-8B model and is fast and stable for scoring.
# ==============================================================================

# ── Cell 9: RAGAS Evaluation ─────────────────────────────────────────────────
# Change 6: New cell — RAGAS scoring with Groq judge
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset
import ast

results_df = pd.read_csv(RESULTS_PATH)
results_df["contexts"] = results_df["contexts"].apply(ast.literal_eval)

ragas_data = Dataset.from_list([
    {
        "question":     row["question"],
        "answer":       row["answer"],
        "contexts":     row["contexts"],
        "ground_truth": row["ground_truth"],
    }
    for _, row in results_df.iterrows()
])

judge_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0))
judge_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=f"sentence-transformers/{ENCODER_MODEL}")
)

print(f"Running RAGAS on {len(ragas_data)} samples (Groq judge)...")
scores = evaluate(
    dataset=ragas_data,
    metrics=[faithfulness, answer_relevancy],
    llm=judge_llm,
    embeddings=judge_emb,
)

faithfulness_score     = scores["faithfulness"]
answer_relevancy_score = scores["answer_relevancy"]
hallucination_rate     = 1 - faithfulness_score

print(f"\nBaseline RAG — Results ({EVAL_SAMPLES} samples)")
print(f"  Faithfulness      : {faithfulness_score:.4f}")
print(f"  Answer Relevancy  : {answer_relevancy_score:.4f}")
print(f"  Hallucination rate: {hallucination_rate:.4f}")

scores_df = scores.to_pandas()
scores_df.to_csv(RESULTS_PATH.replace(".csv", "_ragas.csv"), index=False)
print(f"\nFull per-sample scores saved.")

