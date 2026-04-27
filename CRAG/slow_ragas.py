import os
import ast
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# RAGAS & Langchain
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ==========================================
# 1. SETUP
# ==========================================
load_dotenv()
ENCODER_MODEL = "all-MiniLM-L6-v2"
INPUT_CSV = "evaluacion_final_200.csv"
OUTPUT_CSV = "notas_finales_200.csv"

# ==========================================
# 2. LOCAL LLM SETUP (OLLAMA)
# ==========================================
print("Connecting to local Ollama (Llama 3)...")
local_llm = ChatOllama(model="llama3.2", temperature=0)
judge_llm = LangchainLLMWrapper(local_llm)

judge_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=f"sentence-transformers/{ENCODER_MODEL}")
)

# ==========================================
# 3. LOAD DATA & RESUME LOGIC (¡Aquí está el arreglo!)
# ==========================================
df_in = pd.read_csv(INPUT_CSV)

# --- EL ARREGLO ---
# RAGAS necesita que la columna se llame "contexts" (en plural). 
# Si tu CSV la tiene en singular, la renombramos.
if "context" in df_in.columns and "contexts" not in df_in.columns:
    df_in.rename(columns={"context": "contexts"}, inplace=True)
    print("Corregido: Columna 'context' renombrada a 'contexts'")
# ------------------

# Intentamos convertir la columna 'contexts' de string a lista. 
# Si no existe, dará error, así que nos aseguramos de que está.
if "contexts" in df_in.columns:
    df_in["contexts"] = df_in["contexts"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
else:
    print("¡ERROR FATAL! No se encuentra ni 'context' ni 'contexts' en tu CSV.")
    print(f"Las columnas que tienes son: {df_in.columns.tolist()}")
    exit()

if os.path.exists(OUTPUT_CSV):
    df_out = pd.read_csv(OUTPUT_CSV)
    start_index = len(df_out)
    print(f"Resuming from row {start_index}")
else:
    df_out = pd.DataFrame()
    start_index = 0
    print("Starting new evaluation...")

# ==========================================
# 4. ROW-BY-ROW EVALUATION 
# ==========================================
for i in range(start_index, len(df_in)):
    print(f"Evaluating row {i+1}/{len(df_in)}...")
    row_ds = Dataset.from_pandas(df_in.iloc[[i]])
    
    try:
        results = evaluate(
            dataset=row_ds,
            metrics=[Faithfulness(), AnswerRelevancy()],
            llm=judge_llm,
            embeddings=judge_emb,
            show_progress=False
        )
        
        # Add to dataframe and save IMMEDIATELY
        df_out = pd.concat([df_out, results.to_pandas()], ignore_index=True)
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Row {i+1} saved to {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"Error on row {i+1}: {e}")
        continue 

print("\n--- DONE! Check your notas_finales_200.csv ---")
