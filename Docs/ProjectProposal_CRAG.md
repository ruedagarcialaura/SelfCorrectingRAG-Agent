# Project Proposal – Applied AI and Deep Learning
**ITMD 524 – Spring 2026**

---

## 1. Title & Team Members

**Title:** Self-Corrective RAG with Dynamic Knowledge Augmentation: A Reliable Agentic Approach to Technical Question Answering

**Team Members:**
- Carmen Vazquez Perez de la Cruz
- Laura Rueda Garcia

---

## 2. Abstract

This project addresses the limitations of static Retrieval-Augmented Generation (RAG) systems, which often hallucinate when local information is irrelevant or outdated. We propose a Corrective RAG (CRAG) system that utilizes an intelligent "Router" to classify the relevance of retrieved documents. If local information is found to be insufficient, the agent triggers an external web search to find the correct answer in a single pre-defined source. Our experiment plan focuses on comparing a vanilla RAG baseline against this agentic architecture using metrics for faithfulness and robustness. As optional extra research if time allows, we plan to implement a dynamic feedback loop where validated information from the web is indexed back into our vector database to enable continuous learning.

---

## 3. Problem & Task

The primary problem in current RAG systems is the "blind trust" placed in the retriever; if the vector search returns noisy or irrelevant data, the LLM often generates factually incorrect answers. Our task is to develop an agentic system that can classify document relevance, refine ambiguous knowledge, and generate precise answers by integrating external sources. This is particularly meaningful for technical or fast-moving domains where a static dataset quickly becomes obsolete. By adding a "Self-Collecting" layer, we aim to improve AI safety and reliability.

---

## 4. Dataset / Data Source

Following a Topic-First approach, we evaluated several datasets on Hugging Face to find the most rigorous stress test for our Retrieval Evaluator and Web Search agents.

We are using the **Apple 2024 Environmental Report QA** dataset from Hugging Face, which contains 4,300 technical Q&A pairs focused on a specific topic: renewable energy, carbon footprints, and resource efficiency.

**Link:** https://huggingface.co/datasets/AdamLucek/apple-environmental-report-QA-retrieval

### Data Inspection Plan

Prior to modeling, we plan to conduct the following data inspection steps:

1. Analyze chunk length distribution to calibrate the chunking strategy.
2. Assess domain vocabulary density to anticipate false-positive retrieval rates.
3. Verify label/answer balance across question types.
4. Manually inspect 50–100 samples to identify noisy or ambiguous entries that could bias the Router's evaluation baseline.

---

## 5. Methods Plan

We propose a comparative study between a standard, sequential RAG system and an advanced, state-managed Corrective RAG (CRAG) Multi-Agent architecture orchestrated by LangGraph.

### Baseline Model: Standard RAG Pipeline

**Method:** We will implement a standard, single-pass RAG pipeline. This will use the `all-MiniLM-L6-v2` embedding model to vectorize the dataset, stored in a FAISS vector database. For retrieval, the system performs a standard vector search to find the top-k relevant chunks. These raw, unvalidated chunks are immediately fed to the Generator (Llama 3-8B model) to synthesize the final answer.

**Justification:** This baseline provides an understood reference point to measure the performance, reliability, and hallucination rates of our proposed advanced system.

---

### Improved Model: Agentic CRAG with Continuous Learning (State-Managed)

**Method:** Our advanced approach transforms the passive pipeline into an active agent pipeline managed by LangGraph. The core philosophy is a robust design: no document reaches the generation stage without passing deep validation. We implement a strict single-input/single-output flow with three conditional decision paths (Relevant, Ambiguous, Irrelevant).

A shared **State Dict** (containing query, chunks, scores, and context) is passed between five specialized agents to maintain state. The agents and their roles are:

1. **Retriever Agent** — Tool-based agent that performs FAISS vector search over the local knowledge base using `all-MiniLM-L6-v2` to retrieve the top-k most relevant chunks for a given user query.

2. **Router Agent** — Acts as the decision engine of the pipeline. It uses a Llama 3-8B model to assign a confidence score and classify each retrieved chunk with one of three labels — `RELEVANT`, `AMBIGUOUS`, or `IRRELEVANT` — determining which processing path is activated.

3. **Knowledge Refiner Agent** — Activated exclusively for `AMBIGUOUS` chunks. It uses a Llama 3-8B model to decompose them into atomic sentences, filter out noisy or low-confidence information, and reassemble only the core, relevant content before passing it to the Generator.

4. **Web Search Agent** — Tool-based agent activated if all chunks are `IRRELEVANT`. Rather than performing open-ended web searches, the agent queries a single pre-defined domain.

5. **Dynamic Index Updater Agent** *(Optional)* — Tool-based agent that vectorizes web evidence using `all-MiniLM-L6-v2` and dynamically inserts it back into the FAISS index to close the continuous learning loop.

**Communication Logic:** LangGraph uses Router scores as conditional edges to decide which path is activated. All paths converge at the Generator Node, which only receives validated context.

**Justification:** We chose this agentic CRAG architecture to directly target the known reliability issues (hallucinations, stale data) of standard RAG. It uses advanced deep-learning classifiers (Router Agent) for a robustness design. The dynamic learning loop adds efficiency and scalability.

---

## 6. Experiment Plan

We will conduct the following specific experiments:

- **Benchmarking Accuracy:** Compare the Baseline vs. CRAG on a set of technical questions with answers known to be in the dataset.
- **Robustness/Noise Injection:** Introduce "distractor" documents into the database to see if the Router correctly identifies them as irrelevant.
- **Component Removal Test:** Compare performance with and without the "Knowledge Refinement" step.
- **Evaluation Metrics:** We will use RAGAS (Faithfulness, Answer Relevance) and Precision@k for retrieval.
- **Planned Outputs:** A comparison table of accuracy metrics and an error breakdown chart showing how many hallucinations were prevented by the Router.

---

## 7. Feasibility & Execution Plan

- **Dataset Readiness:** The Hugging Face datasets are public and ready for preprocessing.
- **Code Readiness:** We will start with a LangChain/LlamaIndex framework as our repository foundation.
- **Compute:** We will utilize Google Colab A100 or Chameleon.
- **Timeline:** We aim to have the Baseline and data exploration finished by the Midterm and the full agentic logic completed for the Final Presentation.

| Weeks | Milestone |
|-------|-----------|
| 1–2 | Dataset indexing in FAISS and Llama 3-8B baseline RAG implementation |
| 3–4 | Router Agent development and LangGraph pipeline integration |
| 5–6 | Refiner and Web Search agent deployment; core CRAG pipeline completion |
| 7 | Ablation study and RAGAS benchmarking (Faithfulness/Relevance) |
| 8 | Final debugging, results visualization, and May presentation prep |
| Extension | Dynamic Index Updater and noise injection experiments |

**Risk Management:** If Web Search latency is too high, we will implement a local "Knowledge Cache" to speed up repeated queries. Also, if Llama 3-8B is too slow for the Router, we will use prompt-engineered small models to maintain acceptable latency.

---

## 8. Expected Results & Learning

The **Faithfulness score (RAGAS)** will be the primary metric used to compare the baseline RAG and the CRAG models. Faithfulness measures the percentage of information produced by the model that is derived from the retriever, so if the model produces an answer that does not come directly from the retrieved context, it will show low faithfulness and therefore show high levels of hallucination. In this context, hallucination rates will be used interchangeably with `1 - Faithfulness`, as this makes for a more clear and reproducible measure.

We will show that while using the agentic path has some increase in latency, it yields many more faithful answers than the baseline RAG because the baseline RAG used irrelevant chunks of retrieved information. We will also demonstrate that the Knowledge Refiner contributes to the Faithfulness of the answer by removing noisy chunks of information from generating the final answer. The results will guide how to set the confidence thresholds for the Router so that unnecessary web queries will be avoided while maintaining very high degrees of factuality.

---

## 9. References

LangChain AI. (2024). *LangGraph: Corrective RAG with Local LLMs* [Jupyter Notebook]. GitHub.  
https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_local.ipynb

LangChain AI. (2024). *LangGraph: Agentic RAG* [Jupyter Notebook]. GitHub.  
https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb
