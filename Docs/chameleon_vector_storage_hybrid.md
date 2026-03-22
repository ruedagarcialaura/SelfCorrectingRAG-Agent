# Vector Storage — Hybrid Architecture Spec
# Google Drive (offline embedding) + Chameleon (baseline execution)

## Architecture Overview

```
[Google Colab Pro]                    [Google Drive]                  [Chameleon Node]
  offline_embed.py   →  corpus_index.faiss + id_map.json  →  rclone copy  →  baseline.py
  (embed once)          (persistent storage)                  (each session)
```

- **Google Colab Pro**: runs the offline embedding phase (fast GPU, no setup).
- **Google Drive**: persistent storage for FAISS artifacts — survives across Colab
  sessions and Chameleon leases.
- **Chameleon (RTX 6000)**: downloads artifacts from Drive via rclone and runs the
  full baseline RAG pipeline (Llama 3-8B generator, RAGAS evaluation, long jobs).

---

## Files to Persist

| File | Description |
|---|---|
| `corpus_index.faiss` | FAISS IndexFlatL2 index — used in both baseline and CRAG pipeline |
| `id_map.json` | Dict mapping FAISS integer ID → chunk text + metadata |

Drive path: `MyDrive/crag-vectors/`

---

## Phase 1 — Offline Embedding on Google Colab Pro

Run this once. Mount Drive, embed the corpus, save artifacts.

### `offline_embed.py` (run as a Colab notebook cell)

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs("/content/drive/MyDrive/crag-vectors", exist_ok=True)

DRIVE_DIR     = "/content/drive/MyDrive/crag-vectors"
INDEX_PATH    = f"{DRIVE_DIR}/corpus_index.faiss"
IDMAP_PATH    = f"{DRIVE_DIR}/id_map.json"
ENCODER_MODEL = "all-MiniLM-L6-v2"   # must match Chameleon phase
```

```python
# Cell 2 — Install dependencies
!pip install -q faiss-gpu sentence-transformers datasets
```

```python
# Cell 3 — Embed and save
import faiss
import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Load corpus
dataset = load_dataset("AdamLucek/apple-environmental-report-QA-retrieval")
chunks  = dataset["corpus"]           # adjust split name after inspection
texts   = [c["text"] for c in chunks]

# Encode
encoder    = SentenceTransformer(ENCODER_MODEL)
embeddings = encoder.encode(
    texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
).astype("float32")

# Build FAISS index
d     = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Build ID map
id_map = {
    str(i): {"text": texts[i], "chunk_id": chunks[i].get("id", i)}
    for i in range(len(texts))
}

# Save directly to Drive
faiss.write_index(index, INDEX_PATH)
with open(IDMAP_PATH, "w") as f:
    json.dump(id_map, f)

print(f"Saved {index.ntotal} vectors → {DRIVE_DIR}")
```

At this point `MyDrive/crag-vectors/` contains both artifacts and they are permanent.

---

## Phase 2 — Authenticate rclone with Google Drive on Chameleon

This is a **one-time setup** per Chameleon node. Because the node is headless (no
browser), authentication requires a local machine as an intermediary.

### Step 1 — Install rclone on the Chameleon node

```bash
# On the Chameleon node (via SSH)
curl https://rclone.org/install.sh | sudo bash
rclone version   # verify install
```

### Step 2 — Get an auth token on your LOCAL machine

Your local machine has a browser, so you run the auth flow there.

```bash
# On YOUR LOCAL machine (not the Chameleon node)
# Install rclone if not already present:
# macOS:   brew install rclone
# Windows: winget install Rclone.Rclone
# Linux:   curl https://rclone.org/install.sh | sudo bash

rclone authorize "drive"
```

This opens a browser window. Log in with the Google account that owns the Drive.
After authorizing, rclone prints a token block in the terminal. Copy the entire
JSON block — it looks like:

```
Paste the following into your remote machine --->
{"access_token":"...","token_type":"Bearer","refresh_token":"...","expiry":"..."}
<---End paste
```

### Step 3 — Configure rclone on the Chameleon node

```bash
# On the Chameleon node
rclone config
```

Follow the interactive prompts:

```
n) New remote
name> gdrive
Storage> drive                    # type "drive" or select the number for Google Drive
client_id>                        # leave blank (press Enter)
client_secret>                    # leave blank (press Enter)
scope> 1                          # "Full access to all files"
root_folder_id>                   # leave blank
service_account_file>             # leave blank
Edit advanced config? n
Use auto config? n                # IMPORTANT: select "n" for headless server
```

rclone will then ask:

```
Paste the token you got from your local machine:
```

Paste the full JSON block you copied in Step 2. Then:

```
Configure as a Shared Drive (Team Drive)? n
y) Yes this is OK
```

### Step 4 — Verify the connection

```bash
# List your Drive root to confirm authentication works
rclone ls gdrive:

# List the specific folder
rclone ls gdrive:crag-vectors/
# Expected output:
#   corpus_index.faiss
#   id_map.json
```

---

## Phase 3 — Download Artifacts and Run Baseline on Chameleon

### `setup_node.sh` — run once at the start of each session

```bash
#!/bin/bash
set -e

echo "Downloading artifacts from Google Drive..."
mkdir -p ~/crag-vectors
rclone copy gdrive:crag-vectors/corpus_index.faiss ~/crag-vectors/
rclone copy gdrive:crag-vectors/id_map.json        ~/crag-vectors/
echo "Done. Artifacts at ~/crag-vectors/"
```

```bash
bash setup_node.sh
```

### `baseline.py` — standard RAG pipeline

```python
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

ENCODER_MODEL = "all-MiniLM-L6-v2"   # MUST match the Colab embedding phase
INDEX_PATH    = "crag-vectors/corpus_index.faiss"
IDMAP_PATH    = "crag-vectors/id_map.json"

# Load artifacts — once per process
index   = faiss.read_index(INDEX_PATH)
id_map  = json.load(open(IDMAP_PATH))
encoder = SentenceTransformer(ENCODER_MODEL)

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Return top-k chunks for a query."""
    q_vec = encoder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        entry = id_map[str(idx)]
        results.append({"text": entry["text"], "score": float(dist)})
    return results

# Then pass retrieved chunks directly to Llama 3-8B (no validation in baseline)
```

---

## Dynamic Index Updater — Upload back to Drive (CRAG improved pipeline)

When the Dynamic Index Updater adds web-sourced chunks, re-upload to Drive so the
updated index persists across sessions:

```python
def add_chunks_to_index(new_texts: list[str]):
    new_embeddings = encoder.encode(new_texts, convert_to_numpy=True).astype("float32")
    start_id = index.ntotal
    index.add(new_embeddings)
    for i, text in enumerate(new_texts):
        id_map[str(start_id + i)] = {"text": text, "chunk_id": f"web_{start_id + i}"}

    # Save locally
    faiss.write_index(index, INDEX_PATH)
    with open(IDMAP_PATH, "w") as f:
        json.dump(id_map, f)

    # Re-upload to Drive
    import subprocess
    subprocess.run(["rclone", "copy", INDEX_PATH, "gdrive:crag-vectors/"], check=True)
    subprocess.run(["rclone", "copy", IDMAP_PATH, "gdrive:crag-vectors/"], check=True)
```

---

## Encoder Consistency Constraint

> `ENCODER_MODEL = "all-MiniLM-L6-v2"` must be identical in Colab (Phase 1)
> and Chameleon (Phase 3). This constant must never be changed after the index
> is built without rebuilding the index from scratch and re-uploading.

---

## Key Constraints

- `IndexFlatL2` is used for **both the baseline and the improved CRAG pipeline**.
  215 chunks — exact search, no training required, 100% recall.
- The rclone auth token includes a refresh token and does not expire. It survives
  across Chameleon leases as long as the rclone config file (`~/.config/rclone/rclone.conf`)
  is present on the node. Save this file to Drive or your local machine to avoid
  re-authenticating on every new node.
- Google Drive objects have no size limits relevant to this project.
