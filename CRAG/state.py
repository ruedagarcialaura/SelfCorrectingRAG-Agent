from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our LangGraph workflow.
    """
    question: str          # The original user query
    classification: str    # 'greeting', 'apple_quest', or 'out_of_scope'
    context: List[str]     # The retrieved document chunks from FAISS
    relevance: str         # 'relevant' or 'irrelevant' (determined by your Grader node)
    answer: str            # The final generated response
    steps: List[str]       # A history log of nodes visited for debugging