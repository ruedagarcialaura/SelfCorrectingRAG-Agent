import os
import ast
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# RAGAS Imports
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# Langchain Imports
from langchain_groq import ChatGroq
from langchain_core.outputs import ChatResult
from langchain_community.embeddings import HuggingFaceEmbeddings

import time

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "GROQ_API_KEY not found in environment variables."

ENCODER_MODEL = "all-MiniLM-L6-v2"
INPUT_CSV = "evaluacion_final_200.csv"
OUTPUT_CSV = "notas_finales_200.csv"

# ==========================================
# 2. LOAD DATASET & CHECK PROGRESS
# ==========================================
print(f"Loading generated samples from '{INPUT_CSV}'...")
df = pd.read_csv(INPUT_CSV)

# VITAL: Convert the string representation of lists back into actual Python lists
df["contexts"] = df["contexts"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Check our progress to resume safely
if os.path.exists(OUTPUT_CSV):
    df_out = pd.read_csv(OUTPUT_CSV)
    start_index = len(df_out)
    print(f"\nPrevious file found! You have {start_index} questions already evaluated.")
    if start_index >= len(df):
        print("The evaluation is already 100% complete!")
        exit()
else:
    df_out = pd.DataFrame()
    start_index = 0
    print("\nStarting evaluation from scratch...")

# Slice the dataset to start exactly from where we left off
df_to_evaluate = df.iloc[start_index:]
ragas_data = Dataset.from_pandas(df_to_evaluate)
print(f"Dataset ready. Remaining samples to evaluate: {len(ragas_data)}")

# ==========================================
# 3. LLM & EMBEDDINGS SETUP (Safe Groq)
# ==========================================
class SafeChatGroq(ChatGroq):
    """Custom wrapper to prevent Groq from crashing when RAGAS asks for n > 1 completions."""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        n_requested = kwargs.pop("n", 1)
        kwargs["n"] = 1 # Force n=1
        
        generations = []
        for _ in range(n_requested):     
            res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            generations.extend(res.generations)
        return ChatResult(generations=generations, llm_output=res.llm_output)
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(2) # Small delay to avoid hitting rate limits in rapid succession
        n_requested = kwargs.pop("n", 1)
        kwargs["n"] = 1 # Force n=1
        
        generations = []
        for _ in range(n_requested):
            res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            generations.extend(res.generations)
        return ChatResult(generations=generations, llm_output=res.llm_output)

# Initialize models
print("Initializing LLM and Embeddings...")
safe_groq = SafeChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0)
judge_llm = LangchainLLMWrapper(safe_groq)

judge_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=f"sentence-transformers/{ENCODER_MODEL}")
)

# ==========================================
# 4. PATIENT CONFIGURATION (Rate Limit Fix)
# ==========================================
# max_workers=2: Only evaluate 2 questions simultaneously to avoid spiking API requests.
# timeout=180: Wait up to 3 minutes for the API to respond before failing.
patient_config = RunConfig(
    timeout=180,
    max_retries=10,
    max_workers=2 
)

# ==========================================
# 5. RUN EVALUATION
# ==========================================
print("\nStarting Patient RAGAS Evaluation... (This will take a while, let it run!)")
scores = evaluate(
    dataset=ragas_data,
    metrics=[
        Faithfulness(), 
        AnswerRelevancy()
    ],
    llm=judge_llm,
    embeddings=judge_emb,
    run_config=patient_config,
    raise_exceptions=False # Crucial: If one sample times out, it won't crash the whole script
)

# ==========================================
# 6. CALCULATE & SAVE RESULTS
# ==========================================
score_df = scores.to_pandas()

# MERGE the old saved results with the newly evaluated ones
df_final = pd.concat([df_out, score_df], ignore_index=True)
df_final.to_csv(OUTPUT_CSV, index=False)

# Calculate averages ignoring any NaNs (errors)
faithfulness_score     = df_final["faithfulness"].mean()
answer_relevancy_score = df_final["answer_relevancy"].mean()
hallucination_rate     = 1 - faithfulness_score

print(f"\n--- FINAL EVALUATION RESULTS ({len(df_final)} samples) ---")
print(f"  Faithfulness      : {faithfulness_score:.4f}")
print(f"  Answer Relevancy  : {answer_relevancy_score:.4f}")
print(f"  Hallucination rate: {hallucination_rate:.4f}")

print(f"\nDetailed per-sample scores saved safely to '{OUTPUT_CSV}'")