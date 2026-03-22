# Dataset Working Guide — AdamLucek/apple-environmental-report-QA-retrieval

## 1. Dataset Overview

| Field | Value |
|---|---|
| Source document | Apple's 2024 Environmental Progress Report |
| HuggingFace ID | `AdamLucek/apple-environmental-report-QA-retrieval` |
| License | MIT |
| Total QA pairs | 4,300 |
| Train split | 3,440 rows |
| Validation split | 860 rows |
| Unique text chunks | 215 |
| Questions per chunk | ~20 (synthetic, generated from each chunk) |
| Chunking strategy | Token-based recursive, `chunk_size=800`, `overlap=400` |

### Schema

Each row has exactly three fields:

| Field | Type | Description |
|---|---|---|
| `question` | string | A synthetic question answerable from the chunk |
| `chunk` | string | The source text passage |
| `split` | string | `"train"` or `"validation"` |

The 215 unique chunks appear across both splits. Questions are the unit of variation; chunks repeat ~20 times each.

---

## 2. Loading the Dataset

### With HuggingFace `datasets`

```python
from datasets import load_dataset

ds = load_dataset("AdamLucek/apple-environmental-report-QA-retrieval")
# ds["train"]      → 3,440 rows
# ds["validation"] → 860 rows
```

### With pandas (direct JSON)

```python
import pandas as pd

train_df = pd.read_json(
    "hf://datasets/AdamLucek/apple-environmental-report-QA-retrieval/train.json"
)
val_df = pd.read_json(
    "hf://datasets/AdamLucek/apple-environmental-report-QA-retrieval/validation.json"
)
```

### Extracting the 215 Unique Chunks

```python
# From a datasets object
all_rows = pd.DataFrame(ds["train"]).append(pd.DataFrame(ds["validation"]))
chunks = all_rows["chunk"].drop_duplicates().reset_index(drop=True)
# len(chunks) == 215

# As a list (order matters for FAISS ID alignment)
chunk_list = chunks.tolist()
```

---

## 3. Embedding Setup

| Parameter | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` |
| Output dimension | 384 |
| Output dtype | `float32` |
| Vectors to index | 215 |

> **Constraint:** Use the same model in both the offline build phase and the online query phase. Mixing models produces meaningless similarity scores. See `rag_architecture.md` → *Critical Constraint — Encoder Consistency*.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunk_list, convert_to_numpy=True)  # shape: (215, 384)
embeddings = embeddings.astype("float32")
```

---

## 4. FAISS Index — Why `IndexFlatL2` Is the Right Choice

### Memory math

```
215 vectors × 384 dimensions × 4 bytes/float = 330,240 bytes ≈ 330 KB
```

This is negligible. The entire index fits comfortably in RAM with no compression needed.

### Index selection (from `rag_architecture.md`)

| Dataset size | Recommended index | Notes |
|---|---|---|
| **< 100K vectors** | **`IndexFlatL2`** | Exact search, no training needed |
| 100K – 5M | `IndexIVFFlat` | Requires training, tune `nprobe` |
| 5M – 1B | `IndexIVFPQ` | Compressed, ~90% recall trade-off |

215 vectors is far below the 100K threshold. `IndexFlatL2` gives:

- **Exact nearest-neighbor search** — no approximation, no recall trade-off
- **No training step** — add vectors and query immediately
- **No hyperparameter tuning** — no `nprobe`, no cluster count
- **Deterministic results** — same query always returns the same top-k

There is no reason to use an approximate index at this scale.

---

## 5. Full Build Pipeline

### Install dependencies

```bash
pip install faiss-cpu>=1.7.4 sentence-transformers>=2.2.2 datasets>=2.0.0 numpy>=1.24
```

### Build and save the index

```python
import faiss
import pickle
import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# --- Load dataset and deduplicate chunks ---
ds = load_dataset("AdamLucek/apple-environmental-report-QA-retrieval")
import pandas as pd
all_df = pd.concat([
    pd.DataFrame(ds["train"]),
    pd.DataFrame(ds["validation"])
])
chunk_list = all_df["chunk"].drop_duplicates().reset_index(drop=True).tolist()
assert len(chunk_list) == 215, f"Expected 215 unique chunks, got {len(chunk_list)}"

# --- Embed chunks ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunk_list, convert_to_numpy=True).astype("float32")
# embeddings.shape == (215, 384)

# --- Build FAISS index ---
dim = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
assert index.ntotal == 215, f"Expected 215 vectors in index, got {index.ntotal}"

# --- Save artifacts ---
faiss.write_index(index, "index.faiss")

id_to_chunk = {i: chunk_list[i] for i in range(len(chunk_list))}
with open("id_to_chunk.pkl", "wb") as f:
    pickle.dump(id_to_chunk, f)

config = {
    "embedding_model": "all-MiniLM-L6-v2",
    "dim": dim,
    "chunk_size_tokens": 800,
    "chunk_overlap_tokens": 400,
    "n_chunks": len(chunk_list),
}
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Index built: {index.ntotal} vectors, dim={dim}")
print(f"Index file size: {os.path.getsize('index.faiss') / 1024:.1f} KB")
```

---

## 6. EDA Reference

The `dataset_inspection.ipynb` notebook contains the full EDA. Key things to check and useful one-liners:

### Chunk length distribution

```python
all_df["chunk_len_chars"] = all_df["chunk"].str.len()
all_df["chunk_len_chars"].describe()
# hist
all_df["chunk_len_chars"].hist(bins=30)
```

### Question length distribution

```python
all_df["question_len"] = all_df["question"].str.split().str.len()
all_df["question_len"].describe()
```

### Questions per chunk (should be ~20)

```python
all_df.groupby("chunk")["question"].count().describe()
# min/max/mean — confirm roughly uniform coverage
```

### Sample rows

```python
# Train
ds["train"].select(range(5)).to_pandas()

# Validation
ds["validation"].select(range(5)).to_pandas()
```

### Vocabulary / topic overview

```python
from collections import Counter
import re

words = " ".join(all_df["chunk"].tolist()).lower()
tokens = re.findall(r"\b[a-z]{4,}\b", words)
Counter(tokens).most_common(30)
```

### Split balance check

```python
all_df["split"].value_counts()
# train: 3440  |  validation: 860  (80/20)
```

---

## 7. Query Pipeline

```python
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load artifacts (once at startup)
index = faiss.read_index("index.faiss")
with open("id_to_chunk.pkl", "rb") as f:
    id_to_chunk = pickle.load(f)
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query: str, k: int = 5) -> list[str]:
    """Return top-k chunks most relevant to query."""
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, ids = index.search(query_vec, k)  # ids shape: (1, k)
    return [id_to_chunk[i] for i in ids[0]]

# Example
results = retrieve("What is Apple's renewable energy usage?", k=5)
for i, chunk in enumerate(results, 1):
    print(f"--- Result {i} ---\n{chunk[:300]}\n")
```

> **Note:** The value of `k` will be determined during evaluation. Start with `k=5` and adjust based on LLM context window and retrieval precision experiments.

---

## 8. File Artifacts

| File | Description |
|---|---|
| `index.faiss` | Flat L2 FAISS index, 215 vectors × 384 dim (~330 KB) |
| `id_to_chunk.pkl` | `dict[int, str]` — integer FAISS ID → chunk text |
| `config.json` | Embedding model, dimension, chunk params (for reproducibility) |

---

## 9. Dependencies

```
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
datasets>=2.0.0
numpy>=1.24
```

---

## 10. Verification Checklist

After running the build pipeline, confirm:

- [ ] `index.ntotal == 215`
- [ ] `index.faiss` size on disk is ~330 KB
- [ ] Sample query returns 5 topically relevant chunks
- [ ] `config.json` records the correct model name and dimension

---

## Related Files

| File | Purpose |
|---|---|
| `Docs/rag_architecture.md` | Two-phase pipeline design, index selection guide, encoder consistency constraint |
| `Docs/faiss_architecture.md` | FAISS index type reference |
| `dataset_inspection.ipynb` | Exploratory data analysis of this dataset |
