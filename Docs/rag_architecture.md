

# RAG Architecture — CRAG Project Baseline

## Overview

This document describes the baseline Retrieval-Augmented Generation (RAG) architecture used in the CRAG project. The pipeline is split into two clearly separated phases: an **offline phase** that runs once on Google Colab Pro, and an **online phase** that runs on every user query from Chameleon. Artifacts are persisted to Google Drive as the shared storage layer between both environments.

---

## Architecture Diagram

```
[Google Colab Pro]                  [Google Drive]               [Chameleon Node]
  offline_embed.py  →  corpus_index.faiss + id_map.json  →  rclone  →  baseline.py
  (embed once)            MyDrive/crag-vectors/               (each session)
```

---

## Phase 1 — Offline (Run Once on Google Colab Pro)

The goal of this phase is to embed the full document corpus, build the FAISS index, and persist both artifacts to Google Drive. This step should never need to be repeated unless the corpus changes.

### Steps

1. **Corpus** — The dataset is loaded from Hugging Face (`AdamLucek/apple-environmental-report-QA-retrieval`). The corpus split contains 215 chunks pre-chunked at 800-token size with 400-token overlap.
2. **Encoder** — Each chunk is encoded into a dense float32 vector using `all-MiniLM-L6-v2`.
3. **FAISS Index** — All vectors are added to a `IndexFlatL2` index (exact search, no training required) and saved as `corpus_index.faiss`.
4. **ID Map** — A JSON dictionary mapping each FAISS integer ID to its original chunk text and metadata is saved as `id_map.json`.
5. **Persist to Drive** — Both artifacts are saved directly to `MyDrive/crag-vectors/` and survive indefinitely across Colab sessions and Chameleon leases.

### Outputs

| File | Location | Description |
|---|---|---|
| `corpus_index.faiss` | `MyDrive/crag-vectors/` | `IndexFlatL2` index containing all embedded vectors |
| `id_map.json` | `MyDrive/crag-vectors/` | Dictionary mapping each integer ID to its chunk text and metadata |

---

## Phase 2 — Session Setup (Start of Each Chameleon Session)

Before running the baseline pipeline, the artifacts must be downloaded from Google Drive to the Chameleon node using rclone. This is the only step that requires network transfer and takes only seconds.

### Steps

1. **rclone copy** — Downloads `corpus_index.faiss` and `id_map.json` from `gdrive:crag-vectors/` to `~/crag-vectors/` on the Chameleon node.
2. **Load into memory** — The FAISS index and the ID map are loaded once at process startup and remain in memory for the full session.

### Local Storage Layout on Chameleon

```
~/crag-vectors/
├── corpus_index.faiss     # FAISS IndexFlatL2 — loaded into RAM once per session
└── id_map.json            # int ID → {"text": "...", "chunk_id": "..."}
```

---

## Phase 3 — Online (Every Query)

The goal of this phase is to retrieve the most relevant chunks for a given user query and pass them as context to the LLM. No disk I/O happens at query time — everything runs from memory.

### Steps

1. **Query** — Raw text string from the user.
2. **Encoder** — The query is encoded using `all-MiniLM-L6-v2`, the **exact same model** used in the offline phase.
3. **FAISS Search** — The query vector is compared against all 215 indexed vectors using exact L2 search; the top-k most similar IDs are returned.
4. **ID → Chunk Lookup** — The returned integer IDs are resolved to text chunks using `id_map.json`.
5. **LLM** — The retrieved chunks are concatenated with the query as context and passed to Llama 3-8B.
6. **Answer** — The LLM generates a grounded response.

---

## Critical Constraint — Encoder Consistency

> `all-MiniLM-L6-v2` must be used identically in both the offline phase (Colab) and the online phase (Chameleon).

If the model is changed after the index has been built, query vectors and stored vectors will live in incompatible embedding spaces and similarity search results will be meaningless. Any model change requires discarding the existing index, re-embedding the full corpus on Colab, and re-uploading to Google Drive.

---

## Index Choice Justification

`IndexFlatL2` is used for both the baseline and the improved CRAG pipeline. With only 215 corpus chunks, exact brute-force search is the correct choice: it requires no training, guarantees 100% recall, and the latency difference versus approximate indexes is negligible at this scale.

---

## Dependencies

- `faiss-gpu` (Colab offline phase) / `faiss-cpu` (Chameleon online phase)
- `sentence-transformers`
- `datasets` (Hugging Face)
- `rclone` (Chameleon session setup)
- `numpy`
- `json` (stdlib)
