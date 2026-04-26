import os
import sys
import time
import ast
import pandas as pd
from datasets import load_dataset, Dataset
from dotenv import load_dotenv

# Import the LangGraph compiled app
from main import app 

# Import RAGAS tools
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_core.outputs import ChatResult
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# 1. CONSTANTS & SETUP
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "GROQ_API_KEY not found in environment variables."

if "--samples" in sys.argv:
    idx = sys.argv.index("--samples")
    EVAL_SAMPLES = int(sys.argv[idx + 1])
else:
    EVAL_SAMPLES = 800

HF_DATASET_ID = "AdamLucek/apple-environmental-report-QA-retrieval"
RESULTS_PATH  = "crag_results.csv"
ENCODER_MODEL = "all-MiniLM-L6-v2"

# ==========================================
# 2. LOAD DATASET (Exact same as baseline)
# ==========================================
print(f"Loading dataset '{HF_DATASET_ID}'...")
ds = load_dataset(HF_DATASET_ID, split="validation")
val = pd.DataFrame(ds).drop_duplicates(subset="question")

# 1. Mantenemos el sample ORIGINAL para asegurar que la lista es idéntica a la de tu compañero.
# OJO: Ponemos 800 fijo aquí para que la "mezcla" sea la misma de ayer.
val = val.sample(n=800, random_state=42).reset_index(drop=True)

# 2. AHORA recortamos la lista para coger solo de la pregunta 50 a la 150.
val = val.iloc[150:200]

print(f"Dataset ready: {len(val)} unique questions to evaluate.")

print(f"Running LangGraph evaluation on {EVAL_SAMPLES} samples...")
print(f"Results will be saved to: {RESULTS_PATH}\n")

# ==========================================
# 3. RUN LANGGRAPH EVALUATION
# ==========================================
records = []
start_time = time.time()

for i, row in val.iterrows():
    question = row["question"]
    ground_truth = row["chunk"]
    
    # 1. Pass the question into your LangGraph Agent
    inputs = {"question": question, "steps": []}
    
    try:
        # invoke() runs the whole graph and returns the final state dictionary
        final_state = app.invoke(inputs)
        
        answer = final_state.get("answer", "")
        # If FAISS was skipped, context might be empty, which is valid in CRAG
        contexts = final_state.get("context", []) 
        
    except Exception as e:
        print(f"Error on sample {i}: {e}")
        answer = "Error generating answer."
        contexts = []

    # 2. Store the results
    records.append({
        "question":     question,
        "ground_truth": ground_truth,
        "answer":       answer,
        "contexts":     contexts,
    })
    
    # 3. Logging & Rate Limit Protection
    elapsed = time.time() - start_time
    avg = elapsed / (i + 1)
    remaining = avg * (EVAL_SAMPLES - i - 1)
    
    if (i + 1) % 10 == 0:
        print(f"[{i+1:03d}/{EVAL_SAMPLES}] elapsed: {elapsed/60:.1f} min | remaining: {remaining/60:.1f} min")

    # Save checkpoints every 50 rows just like the baseline
    if (i + 1) % 50 == 0:
        pd.DataFrame(records).to_csv(RESULTS_PATH, index=False)
    
    # Tiny sleep to avoid Groq API Rate Limits (Tokens Per Minute / Requests Per Minute)
    time.sleep(0.5) 

# Final save
pd.DataFrame(records).to_csv(RESULTS_PATH, index=False)
print(f"\nGraph Generation complete. {len(records)} samples saved to {RESULTS_PATH}\n")


# ==========================================
# 4. RAGAS SCORING (Exact same judge setup)
# ==========================================
print("Starting RAGAS Evaluation...")

# Load the CSV we just made
results_df = pd.read_csv(RESULTS_PATH)
# Ensure lists are read properly from the CSV
results_df["contexts"] = results_df["contexts"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

ragas_data = Dataset.from_list([
    {
        "question":     row["question"],
        "answer":       row["answer"],
        "contexts":     row["contexts"],
        "ground_truth": row["ground_truth"],
    }
    for _, row in results_df.iterrows()
])

class SafeChatGroq(ChatGroq):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # 1. Miramos cuántas pide RAGAS (si no dice nada, asumimos 1)
        n_requested = kwargs.pop("n", 1)
        
        # 2. ¡EL TRUCO DEFINITIVO! Le decimos a Groq por la fuerza que SOLO haga 1
        kwargs["n"] = 1 
        
        generations = []
        for _ in range(n_requested):     
            res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            generations.extend(res.generations)
        return ChatResult(generations=generations, llm_output=res.llm_output)
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        n_requested = kwargs.pop("n", 1)
        
        # Lo forzamos también en las llamadas asíncronas
        kwargs["n"] = 1 
        
        generations = []
        for _ in range(n_requested):
            res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            generations.extend(res.generations)
        return ChatResult(generations=generations, llm_output=res.llm_output)

safe_groq = SafeChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0)

judge_llm = LangchainLLMWrapper(safe_groq)
judge_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=f"sentence-transformers/{ENCODER_MODEL}")
)

print(f"Running RAGAS on {len(ragas_data)} samples (Groq judge)...")
scores = evaluate(
    dataset=ragas_data,
    metrics=[
        Faithfulness(), 
        AnswerRelevancy()
    ],
    llm=judge_llm,
    embeddings=judge_emb,
)

# FIX 2: Convertimos a Pandas para calcular la media (promedio) de forma segura
scores_df = scores.to_pandas()

# Sacamos la media omitiendo los posibles errores
faithfulness_score     = scores_df["faithfulness"].mean()
answer_relevancy_score = scores_df["answer_relevancy"].mean()
hallucination_rate     = 1 - faithfulness_score

print(f"\nCRAG Agent — Results ({EVAL_SAMPLES} samples)")
print(f"  Faithfulness      : {faithfulness_score:.4f}")
print(f"  Answer Relevancy  : {answer_relevancy_score:.4f}")
print(f"  Hallucination rate: {hallucination_rate:.4f}")

# Save final scores
scores_df.to_csv(RESULTS_PATH.replace(".csv", "_ragas.csv"), index=False)
print(f"\nFull per-sample scores saved to {RESULTS_PATH.replace('.csv', '_ragas.csv')}")