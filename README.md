# Self-Correcting RAG Agent with Continuous Learning

This project implements an advanced Multi-Agent Corrective RAG (CRAG) system designed to eliminate hallucinations and handle out-of-distribution queries through automated validation and real-time web augmentation. The system acts as an evaluate-before-generating pipeline that improves reliability in domain-specific technical QA. 

The primary knowledge base is built upon the 2024 Apple Environmental Progress Report dataset, comprising 3,440 technical Q&A pairs.

## Architecture


The CRAG pipeline operates as a directed acyclic graph (DAG) built with LangGraph, transitioning from a linear "blind" retriever to a "retrieve-grade-refine" model. It utilizes a hybrid inference strategy for high-speed routing and localized generation.

* **Router Agent:** Uses the Groq API as a deterministic classifier (temperature 0) to categorize queries as *greeting*, *apple_quest*, or *out_of_scope* before retrieving data.
* **Retriever:** A local FAISS index (`faiss.IndexFlatL2`) that performs exact brute-force L2 search. Chunks are processed using a `RecursiveCharacterTextSplitter` (200 tokens, 10% overlap) and embedded via `all-MiniLM-L6-v2`.
* **Grader Agent:** Powered by Groq, this strict filter analyzes the semantic relevance of retrieved documents against the original query. If the context is deemed irrelevant, it triggers the web search node.
* **Web Search Node:** A dynamic fallback activated by the Grader to perform real-time searches via DuckDuckGo, replacing failed FAISS context with updated external information.
* **Generator Agent:** A custom-deployed `Llama-3.1-8B-Instruct` model synthesizes the final response, strictly anchored to the validated local or web-sourced context to prevent hallucinations.

## Performance & Results

The system was evaluated against 800 deduplicated queries using the RAGAS framework alongside a plain RAG baseline. The corrective architecture successfully reduced hallucinations when faced with out-of-knowledge queries.

| Metric | Plain RAG | Corrective RAG (CRAG) |
| :--- | :--- | :--- |
| **Faithfulness** | 0.8868 | 0.9002 |
| **Answer Relevancy** | 0.7389 | 0.8187 |
| **Hallucination Rate** | 0.1132 | 0.0998 |

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

1. Environment Setup: Clone the repository and install the dependencies in `requirements.txt`. If you are running this in Google Colab, ensure your runtime is set to a T4 GPU.
   ```bash
   pip install -r requirements.txt
   ```
2. Build the FAISS Index: Run the indexing script first. This will download the dataset from Hugging Face, apply the token-aware chunking strategy, generate the sentence embeddings, and save the FAISS index locally.

3. Run the Baseline RAG: Execute the baseline_rag.ipynb notebook to run the plain RAG on the 800 test questions. You will obtain the RAGAS score (faithfulness, answer relevancy, and hallucination rate) in the results.

4. Run the LangGraph Pipeline: Execute the main script to start the agentic workflow. You will need to provide a Hugging Face API token with access to the Meta Llama-3.1-8B-Instruct repository. 

5. Run the Evaluation: To reproduce the evaluation metrics, run the RAGAS evaluation script.

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

```