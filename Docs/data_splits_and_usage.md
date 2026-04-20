# Data Splits & Usage — Corpus vs. Evaluation

**Project:** Self-Corrective RAG (CRAG) — ITMD 524, Spring 2026  
**Dataset:** [AdamLucek/apple-environmental-report-QA-retrieval](https://huggingface.co/datasets/AdamLucek/apple-environmental-report-QA-retrieval)

---

## 1. Dataset Structure

The dataset is derived from **Apple's 2024 Environmental Progress Report** and contains synthetic question–answer pairs grounded in the source document.

| Field | Value |
|---|---|
| Total QA pairs | 4,300 |
| Train split | 3,440 rows |
| Validation split | 860 rows |
| Unique text chunks (across both splits) | 215 |
| Questions per chunk | ~20 (synthetically generated) |

Each row has two columns: `question` (a synthetic question) and `chunk` (the source text passage that answers it). The 215 unique chunks appear across **both** splits — questions are the unit of variation, chunks repeat ~20 times each.

---

## 2. What Data is Used for the Corpus (Retrieval Index)

### Source
- **Both splits** (train + validation) are merged and deduplicated on the `chunk` column.
- This yields **215 unique source text passages**.

### Processing
- Each of the 215 source chunks is **re-split** using a token-aware `RecursiveCharacterTextSplitter`:
  - `chunk_size = 200 tokens`
  - `chunk_overlap = 20 tokens`
  - Length function: the `all-MiniLM-L6-v2` tokenizer (includes `[CLS]` and `[SEP]` tokens)
- This produces **1,158 sub-chunks**, each guaranteed to fit within the encoder's 256-token context window.

### Embedding & Indexing
- Each of the 1,158 sub-chunks is encoded into a 384-dimensional float32 vector using `all-MiniLM-L6-v2`.
- Vectors are stored in a **FAISS `IndexFlatL2`** index (exact brute-force L2 search, no training required).
- An `id_map.json` maps each FAISS integer ID → sub-chunk text + metadata (original chunk index, chunk_id).

### Artifacts (on Google Drive → Chameleon)
| File | Size | Description |
|---|---|---|
| `corpus_index.faiss` | ~1.7 MB | FAISS IndexFlatL2 — 1,158 vectors × 384 dim |
| `id_map.json` | ~870 KB | Maps FAISS int ID → {text, chunk_id, original_chunk_idx} |
| `config.json` | ~1 KB | Build config (model, chunking params, counts, date) |

### Key Point
> The corpus includes **all 215 unique chunks** from **both train and validation splits**. The entire Apple Environmental Report content is indexed. There is no train/test split at the corpus level — every passage is available for retrieval.

---

## 3. What Data is Used for Evaluation

### Source
- The **validation split only** (860 rows) is used for evaluation.
- After deduplication on `question`, a **random sample of 800 questions** is drawn (`random_state=42`).

### Why Validation Only
- The train split's questions could be used to tune retrieval parameters or router thresholds.
- The validation split provides unseen questions for unbiased evaluation of the full pipeline.
- Questions are the unit being evaluated (not chunks), so using validation questions ensures the pipeline's *generation* quality is tested on held-out inputs.

### Evaluation Procedure
For each of the 800 sampled questions:
1. **Retrieve** top-5 sub-chunks from the FAISS index using the question as query.
2. **Generate** an answer using the local Llama 3.1-8B-Instruct model (greedy decoding).
3. **Record**: question, ground_truth chunk, generated answer, and retrieved context list.

### RAGAS Scoring
After all 800 answers are generated, RAGAS evaluates them using:

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Is the generated answer grounded in the retrieved context? (Higher = less hallucination) |
| **Answer Relevancy** | Is the answer relevant to the question asked? |
| **Hallucination Rate** | Defined as `1 - Faithfulness` |

The RAGAS judge uses **Groq** (`llama-3.1-8b-instant` via API), not the local model — wrapping a local HuggingFace model as a RAGAS judge is unreliable, and Groq provides the same Llama 3.1-8B with faster, more stable scoring.

---

## 4. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│              HuggingFace Dataset                                 │
│   AdamLucek/apple-environmental-report-QA-retrieval              │
│                                                                  │
│   ┌──────────────────┐     ┌──────────────────┐                  │
│   │   Train Split     │     │  Validation Split │                 │
│   │   3,440 rows      │     │  860 rows         │                 │
│   │   (questions +    │     │  (questions +     │                 │
│   │    chunks)        │     │   chunks)         │                 │
│   └────────┬─────────┘     └────────┬──────────┘                 │
│            │                        │                             │
│            └────────┬───────────────┘                             │
│                     │                                             │
│                     ▼                                             │
│          ┌──────────────────┐                                    │
│          │ Deduplicate on   │                                    │
│          │ chunk column     │                                    │
│          │ → 215 unique     │                                    │
│          │   text passages  │                                    │
│          └────────┬─────────┘                                    │
│                   │                                               │
│                   ▼                                               │
│    ┌──────────────────────────┐                                  │
│    │ Re-chunk (200 tokens,    │                                  │
│    │ 20 overlap) → 1,158      │                                  │
│    │ sub-chunks               │                                  │
│    └────────┬─────────────────┘                                  │
│             │                                                     │
│             ▼                                                     │
│    ┌──────────────────────────┐                                  │
│    │ Embed with               │                                  │
│    │ all-MiniLM-L6-v2         │                                  │
│    │ → 1,158 × 384 vectors    │                                  │
│    └────────┬─────────────────┘                                  │
│             │                                                     │
│             ▼                                                     │
│    ┌──────────────────────────┐         ┌────────────────────┐   │
│    │ FAISS IndexFlatL2        │         │  CORPUS             │   │
│    │ corpus_index.faiss       │◄────────│  (used for          │   │
│    │ id_map.json              │         │   retrieval)        │   │
│    │ config.json              │         └────────────────────┘   │
│    └──────────────────────────┘                                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│              Evaluation Pipeline                                 │
│                                                                  │
│    Validation Split (860 rows)                                   │
│         │                                                        │
│         ▼                                                        │
│    Deduplicate on question                                       │
│         │                                                        │
│         ▼                                                        │
│    Random sample 800 questions (seed=42)                         │
│         │                                                        │
│         ▼                                                        │
│    For each question:                                            │
│      ┌─────────────┐                                             │
│      │  retrieve()  │ → top-5 sub-chunks from FAISS index        │
│      └──────┬──────┘                                             │
│             │                                                    │
│             ▼                                                    │
│      ┌─────────────┐                                             │
│      │  generate()  │ → answer via local Llama 3.1-8B            │
│      └──────┬──────┘                                             │
│             │                                                    │
│             ▼                                                    │
│    Save: question, ground_truth, answer, contexts                │
│         │                                                        │
│         ▼                                                        │
│    RAGAS Scoring (Groq judge):                                   │
│      • Faithfulness                                              │
│      • Answer Relevancy                                          │
│      • Hallucination Rate = 1 − Faithfulness                     │
│                                                                  │
│    Output files:                                                 │
│      ~/baseline_results.csv       (raw Q/A/context)              │
│      ~/baseline_results_ragas.csv (per-sample RAGAS scores)      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Summary Table

| Aspect | Corpus (Index) | Evaluation |
|---|---|---|
| **Data source** | Both splits (train + val) | Validation split only |
| **Unit** | Text chunks (passages) | Questions |
| **Deduplication** | On `chunk` column → 215 unique | On `question` column |
| **Sample size** | All 215 chunks → 1,158 sub-chunks | 800 questions (random, seed=42) |
| **Processing** | Chunked → embedded → FAISS index | retrieve → generate → RAGAS score |
| **Output** | FAISS index + id_map + config | Results CSV + RAGAS scores CSV |
| **Model used** | `all-MiniLM-L6-v2` (encoder) | Llama 3.1-8B (local generator) + Groq (RAGAS judge) |

---

## 6. Why This Separation Matters

1. **No data leakage in retrieval:** All 215 chunks are indexed because the retriever needs access to the full knowledge base. The corpus *is* the knowledge base — withholding documents would make the RAG system artificially weaker.

2. **Clean evaluation on unseen questions:** Only validation-split questions are used for scoring. Even though the underlying chunks overlap (the same chunk can generate both train and validation questions), the *questions* are distinct — ensuring the pipeline's generation and retrieval quality is tested on inputs it hasn't been optimized for.

3. **Reproducibility:** The 800-sample evaluation uses `random_state=42` for deterministic sampling. Combined with greedy decoding (`do_sample=False`) in the generator, results should be fully reproducible across runs on the same hardware.

---

## Related Files

| File | Purpose |
|---|---|
| [baseline_rag_chameleon.ipynb](../baseline_rag_chameleon.ipynb) | Chameleon notebook implementing the evaluation pipeline |
| [baseline_rag_localversion.ipynb](../baseline_rag_localversion.ipynb) | Original Colab/Groq version of the baseline |
| [offline_embed.ipynb](../offline_embed.ipynb) | Corpus build pipeline (run on Google Colab) |
| [dataset_guide.md](dataset_guide.md) | Full dataset schema, loading, and EDA reference |
| [rag_architecture.md](rag_architecture.md) | Two-phase pipeline design and encoder consistency constraint |
| [eda_conclusions.md](eda_conclusions.md) | EDA findings that motivated the architecture |
