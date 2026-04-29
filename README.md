# Self-Correcting RAG Agent with Continuous Learning

This project implements an advanced Multi-Agent Corrective RAG (CRAG) system designed to eliminate hallucinations and handle out-of-distribution queries through automated validation and real-time web augmentation.

## Architecture

The system is orchestrated using LangGraph to manage a stateful workflow between specialized agents:

1. Retriever Agent: Fetches the top-k document chunks from a FAISS vector database.
2. Router Agent (Evaluator): Uses a local Llama 3.1-8B model to classify retrieved chunks as relevant, ambiguous, or irrelevant.
3. Knowledge Refiner: Processes ambiguous data by decomposing chunks into atomic facts to remove noise.
4. Web Search Agent: Triggers an external search via DuckDuckGo when local knowledge is deemed irrelevant or out of scope.
5. Dynamic Index Updater: Vectorizes new web findings and updates the local FAISS index, closing the continuous learning loop.

## Dataset and Preprocessing

Source: Apple Environmental Progress Report 
Link: https://huggingface.co/datasets/AdamLucek/apple-environmental-report-QA-retrieval

To build the knowledge base, the original source passages were processed into a FAISS vector index. The text was re-chunked into sub-chunks of 200 tokens with a 20-token overlap. This ensures compatibility with the 256-token limit of our embedding model, all-MiniLM-L6-v2, preventing information loss during retrieval. 

## Hardware Requirements

Running the full pipeline and the local Llama 3.1-8B model requires a GPU. 
For basic execution, a minimum of 16GB VRAM is recommended (e.g., an NVIDIA T4 via Google Colab).
For our large-scale evaluation, we utilized a Chameleon cluster with 4x NVIDIA Tesla P100 GPUs, leveraging 4-bit quantization and model parallelism to distribute the VRAM load.

## Dependencies

The project relies on Python 3.10+ and the following core libraries:
* langchain
* langgraph
* faiss-cpu
* torch
* transformers
* accelerate
* bitsandbytes
* ragas
* sentence-transformers
* duckduckgo-search
* pandas

## Reproduction Steps

1. Environment Setup
Clone the repository and install the dependencies in requierements.txt. If you are running this in Google Colab, ensure your runtime is set to a T4 GPU. 

```bash
pip install -r requirements.txt
```

3. Build the FAISS Index
Run the indexing script first. This will download the dataset from Hugging Face, apply the token-aware chunking strategy, generate the sentence embeddings, and save the FAISS index locally.

4. Run the baseline RAG
Execute the baseline_rag.ipynb notebook to run the plain RAG on the 800 test questions. You will obtain the RAGAS score (faithfullness, answer relevancy and hallucination rate) in the results.
 
5. Run the LangGraph Pipeline
Execute the main script to start the agentic workflow. You will need to provide a Hugging Face API token with access to the Meta Llama 3.1-8B-Instruct repository. The system will accept queries, retrieve documents, grade their relevance, and fall back to web search if necessary.

6. Run the Evaluation
To reproduce the evaluation metrics, run the RAGAS evaluation script. The script is configured to evaluate rows sequentially to prevent timeout errors and resource exhaustion.

## Repository Structure

```text
.
├── baseline/                     # Traditional RAG implementation (Baseline)
│   ├── baseline_rag.ipynb        # Notebook for baseline RAG execution
│   ├── offline_embed.ipynb       # Notebook for offline embedding generation
│   └── local_testing/            # Scripts for testing the baseline pipeline locally
├── CRAG/                         # Main directory for Corrective RAG architecture
│   ├── agents.py                 # Logic for specialized agents (Router, Grader, Generator)
│   ├── tools.py                  # External tools (Web Search fallback via DuckDuckGo)
│   ├── state.py                  # LangGraph Graph State definition
│   ├── slow_ragas.py             # Sequential RAGAS evaluation to prevent timeouts
│   ├── main.py                   # Main entry point for the CRAG pipeline
│   └── evaluate_crag.py          # Full evaluation suite for the CRAG system
├── crag-vectors/                 # Vector database and configuration files
│   ├── config.json               # Index configuration and parameters
│   ├── corpues_index.faiss       # Persisted FAISS vector index
│   └── id_map.json               # Mapping of vector IDs to document chunks
├── crag_env/                     # Virtual environment directory
├── EDA/                          # Exploratory Data Analysis
│   ├── images/                   # Generated plots and visualization assets
│   └── dataset_inspection.ipynb  # Notebook for initial data exploration
├── .env                          # Environment variables and API tokens
├── .gitignore                    # Git exclusion rules
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── setup_env.sh                  # Environment setup script
