# Baseline RAG — Jupyter Notebook on Chameleon (SSH Tunnel)

## How This Works

```
[Chameleon Node — GPU]          [SSH Tunnel]          [Your Browser]
  Jupyter server          ←→   localhost:8888   ←→   notebook UI
  baseline_rag.ipynb
  corpus_index.faiss
  id_map.json
```

The notebook runs **physically on the Chameleon node**. It has direct access to
the GPU, the FAISS artifacts, and the `crag` conda environment. Your browser is
just a UI — all computation happens on the node.

---

## Prerequisites

Before opening the notebook:

1. Chameleon lease is **ACTIVE** and the instance is running.
2. `setup_env.sh` has been run on the node (`conda activate crag` works).
3. `setup_node.sh` has been run — artifacts are at `~/crag-vectors/`.

Verify both:
```bash
ssh -i ~/.ssh/chameleon_key cc@<FLOATING_IP>
conda activate crag && python -c "import faiss; print('faiss ok')"
ls ~/crag-vectors/          # should show corpus_index.faiss and id_map.json
```

---

## Step 1 — Start Jupyter on the Chameleon Node

```bash
# On the Chameleon node (via SSH)
conda activate crag
jupyter notebook --no-browser --port=8888

# The terminal will print something like:
#   http://127.0.0.1:8888/?token=abc123def456...
# Copy the full token value — you will need it in Step 3.
```

Leave this terminal open for the entire session.

---

## Step 2 — Open the SSH Tunnel (local machine)

Open a **new terminal** on your local machine (do not close the SSH session above):

```bash
ssh -i ~/.ssh/chameleon_key -N -L 8888:localhost:8888 cc@<FLOATING_IP>
```

This terminal will appear to hang — that is correct, it is forwarding the port.
Leave it open for the entire session.

---

## Step 3 — Open the Notebook in Your Browser

```
http://localhost:8888/?token=<YOUR_TOKEN>
```

You are now inside the Jupyter interface running on the Chameleon node.

---

## Step 4 — Notebook Structure: `baseline_rag.ipynb`

Create this notebook on the node. The notebook must implement the following
cells in order. The implementation code lives in the baseline source files —
do not duplicate logic here.

| Cell | Purpose |
|---|---|
| 1 | Imports and constants (`ENCODER_MODEL`, `INDEX_PATH`, `IDMAP_PATH`, `TOP_K`) |
| 2 | Load FAISS index and id_map — smoke test with `index.ntotal` |
| 3 | `retrieve(query, top_k)` function |
| 4 | Load Llama 3-8B generator with `device_map="auto"` |
| 5 | `generate(query, context_chunks)` function |
| 6 | `baseline_rag(query)` — full pipeline: retrieve → generate |
| 7 | Batch evaluation over the QA dataset with RAGAS |
| 8 | Save results to `baseline_results.csv` |

---

## Keeping the Session Alive

For long evaluation runs, use `tmux` so the job survives if your connection drops:

```bash
# On the node, before starting Jupyter:
tmux new -s crag
conda activate crag
jupyter notebook --no-browser --port=8888

# Detach: Ctrl+B then D
# Reattach later:
tmux attach -t crag
```

---

## End of Session

```bash
# Save results to Drive before the lease expires
rclone copy ~/baseline_results.csv gdrive:crag-vectors/

# Shut down Jupyter: File → Shut Down
# Close the SSH tunnel: Ctrl+C in the tunnel terminal
```

---

## Key Constraints

- The notebook kernel runs inside the `crag` conda env on the node.
- `ENCODER_MODEL` must be identical to the one used in `offline_embed.py`
  (Colab phase). Both use `"all-MiniLM-L6-v2"`.
- The baseline pipeline has **no retrieval validation** — chunks go directly
  to the generator. That is intentional: the CRAG pipeline adds the
  Retrieval Evaluator/Router on top of this.
- `faiss.IndexFlatL2` is used for both the baseline and the CRAG pipeline.
  215 chunks — exact search, no training required, 100% recall.
