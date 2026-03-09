# Self-Correcting RAG Agent with Continuous Learning

This project implements an advanced **Multi-Agent Corrective RAG (CRAG)** system designed to eliminate hallucinations and handle out-of-distribution queries through automated validation and real-time web augmentation.

##  Architecture
The system is orchestrated using **LangGraph** to manage a stateful workflow between five specialized agents:

1.  **Retriever Agent**: Fetches the top-k document chunks from a **FAISS** vector database.
2.  **Router Agent (Evaluator)**: Uses a **Llama 3-8B** model to classify retrieved chunks as `RELEVANT`, `AMBIGUOUS`, or `IRRELEVANT`.
3.  **Knowledge Refiner**: Processes `AMBIGUOUS` data by decomposing chunks into atomic facts to remove noise.
4.  **Web Search Agent**: Triggers an external search (Tavily/DuckDuckGo) when local knowledge is deemed `IRRELEVANT`.
5.  **Dynamic Index Updater**: Vectorizes new web findings and updates the local FAISS index, closing the **continuous learning loop**.

Dataset: https://huggingface.co/datasets/AdamLucek/apple-environmental-report-QA-retrieval
