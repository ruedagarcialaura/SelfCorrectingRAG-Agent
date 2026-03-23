# EDA Conclusions & Design Decisions

## 1. Chunk Size Definition

The most common mistake in RAG pipelines is splitting text arbitrarily. The EDA informs how to fragment the documents correctly.

**What to examine:** The distribution of words (`word_count`) and characters (`char_count`) across corpus chunks.

**Decision for FAISS:** The corpus average is ~480 words per chunk, but the embedding model (`all-MiniLM-L6-v2`) has a maximum context window of 256 tokens. Feeding full chunks without re-splitting would silently truncate roughly half the content before encoding, degrading retrieval quality.

**Notebook entry:**
> "Since the majority of chunks exceed 256 tokens, we apply a `RecursiveCharacterTextSplitter` with `chunk_size=800` characters and `overlap=100` characters to preserve semantic continuity across boundaries."

---

## 2. Estimating the Critical Similarity Threshold (Router Agent Threshold)

Before the Router Agent can label a retrieved chunk as "irrelevant", we need a baseline understanding of how similar the corpus documents are to each other.

**What to examine:** The semantic similarity heatmap across chunk pairs.

**Decision for FAISS / Router:** If the heatmap shows dense dark-blue regions (cosine similarity > 0.8) between topically distinct chunks, the retriever will frequently surface near-duplicate or off-topic results. This high overlap is the quantitative justification for including a Router Agent that re-scores and filters candidates before passing them to the generator.

**Notebook entry:**
> "High intra-corpus semantic overlap (average pairwise similarity > 0.75) justifies a Router Agent to disambiguate retrieved chunks and filter out topically irrelevant results before generation."

---

## 3. Detecting the Vocabulary Gap (Knowledge Refiner Justification)

Users often phrase questions differently from how the source document expresses the same concept — this is the vocabulary mismatch problem.

**What to examine:** Lexical overlap between question tokens and answer tokens (e.g., Jaccard overlap or shared unigram ratio).

**Decision:** If the average lexical overlap is low (< 0.3), a simple RAG pipeline will fail because sparse keyword matching between the query and retrieved chunk is unreliable. A Knowledge Refiner Agent is needed to bridge this gap — either by rewriting the retrieved context into a form more aligned with the query, or by filtering out chunks whose surface form is misleading despite semantic proximity.

**Notebook entry:**
> "Low average lexical overlap between questions and their ground-truth answer passages (overlap < 0.3) indicates a significant vocabulary gap. This motivates the inclusion of a Knowledge Refiner Agent to post-process retrieved chunks before generation, reducing noise introduced by terminological mismatches."
