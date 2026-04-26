import os
import ast
import time
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_core.outputs import ChatResult
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# 1. SETUP
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "GROQ_API_KEY not found in environment variables."

ENCODER_MODEL = "all-MiniLM-L6-v2"
INPUT_CSV = "evaluacion_final_200.csv"
OUTPUT_CSV = "notas_finales_200.csv"

# ==========================================
# 2. SAFE LLM (WITH TURTLE MODE 🐢)
# ==========================================
class SafeChatGroq(ChatGroq):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(5) # Wait 5 seconds for each API request
        n_requested = kwargs.pop("n", 1)
        kwargs["n"] = 1
        
        generations = []
        for _ in range(n_requested):     
            res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            generations.extend(res.generations)
        return ChatResult(generations=generations, llm_output=res.llm_output)
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(5) # Wait 5 seconds for each API request
        n_requested = kwargs.pop("n", 1)
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

# ==========================================
# 3. RESUME AND AUTO-SAVE SYSTEM
# ==========================================
print(f"Loading questions from '{INPUT_CSV}'...")
df_in = pd.read_csv(INPUT_CSV)
df_in["contexts"] = df_in["contexts"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Check where we left off
if os.path.exists(OUTPUT_CSV):
    df_out = pd.read_csv(OUTPUT_CSV)
    start_index = len(df_out)
    print(f"\nPrevious file found! You have {start_index} questions already evaluated.")
    if start_index >= len(df_in):
        print("The evaluation is already 100% complete!")
        exit()
else:
    df_out = pd.DataFrame()
    start_index = 0
    print("\nStarting evaluation from scratch...")

# ==========================================
# 4. ROW BY ROW LOOP
# ==========================================
print(f"We will evaluate from row {start_index + 1} to {len(df_in)}.\n")

for i in range(start_index, len(df_in)):
    # Take ONLY 1 row
    row = df_in.iloc[[i]]
    ds_row = Dataset.from_pandas(row)
    
    try:
        # Evaluate that single row
        score = evaluate(
            dataset=ds_row,
            metrics=[Faithfulness(), AnswerRelevancy()],
            llm=judge_llm,
            embeddings=judge_emb,
            raise_exceptions=False,
            show_progress=False # Disabled to avoid terminal clutter
        )
        
        # Convert to dataframe and append to what we already have
        score_df = score.to_pandas()
        df_out = pd.concat([df_out, score_df], ignore_index=True)
        
        # AUTO-SAVE TO DISK!
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"[{i+1:03d}/{len(df_in)}]  Row evaluated and SAVED.")
        
    except Exception as e:
        print(f"\n Groq crashed on row {i+1}: {e}")
        print("Don't worry. Everything before this is securely saved in the CSV.")
        print("Run the script again in a couple of minutes to continue from this exact spot.")
        break # Safely stop the script

# If we reach the end, calculate the global score
if len(df_out) == len(df_in):
    f_score = df_out["faithfulness"].mean()
    ar_score = df_out["answer_relevancy"].mean()
    hr_score = 1 - f_score
    print(f"\n--- EVALUATION COMPLETED! ---")
    print(f"  Faithfulness      : {f_score:.4f}")
    print(f"  Answer Relevancy  : {ar_score:.4f}")
    print(f"  Hallucination rate: {hr_score:.4f}")