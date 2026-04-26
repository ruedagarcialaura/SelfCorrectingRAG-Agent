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
# 2. LOAD DATASET (200 Samples)
# ==========================================
print(f"Loading generated samples from '{INPUT_CSV}'...")
df = pd.read_csv(INPUT_CSV)

# VITAL: Convert the string representation of lists back into actual Python lists
df["contexts"] = df["contexts"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Convert to HuggingFace Dataset
ragas_data = Dataset.from_pandas(df)
print(f"Dataset ready. Total samples to evaluate: {len(ragas_data)}")

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
scores_df = scores.to_pandas()

# Calculate averages ignoring any NaNs (errors)
faithfulness_score     = scores_df["faithfulness"].mean()
answer_relevancy_score = scores_df["answer_relevancy"].mean()
hallucination_rate     = 1 - faithfulness_score

print(f"\n--- FINAL EVALUATION RESULTS (200 samples) ---")
print(f"  Faithfulness      : {faithfulness_score:.4f}")
print(f"  Answer Relevancy  : {answer_relevancy_score:.4f}")
print(f"  Hallucination rate: {hallucination_rate:.4f}")

# Save the detailed scores
scores_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDetailed per-sample scores saved safely to '{OUTPUT_CSV}'")