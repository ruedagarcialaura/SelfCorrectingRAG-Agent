import os
from groq import Groq
from dotenv import load_dotenv
from state import GraphState

# We will eventually import your teammate's baseline functions here
# from tools import retrieve, generate 

# Setup Groq Client
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.1-8b-instant"

def router_node(state: GraphState):
    """
    Node 1: Analyzes the question and categorizes it before any retrieval happens.
    """
    print("---NODE: ROUTER---")
    question = state["question"]
    steps = state.get("steps", [])

    prompt = f"""
    You are an expert at classifying user queries.
    Analyze the following user question and classify it into exactly ONE of these three categories:
    1. 'greeting': If the user is just saying hello or asking who you are.
    2. 'apple_quest': If the question is technical or informative regarding Apple Inc., its sustainability, or its environmental report.
    3. 'out_of_scope': For any other topic (e.g., sports, politics, recipes).

    Respond with ONLY the category word. Do not add punctuation or extra text.
    Question: {question}
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0 # Temperature 0 ensures the output is strictly the category word
    )
    
    classification = response.choices[0].message.content.strip().lower()
    steps.append("router_node")
    
    return {
        "classification": classification,
        "steps": steps
    }

def retrieval_node(state: GraphState):
    """
    Node 2: Connects to your teammate's FAISS retrieve tool.
    """
    print("---NODE: RETRIEVAL---")
    question = state["question"]
    steps = state.get("steps", [])
    
    # FUTURE CODE (Integration):
    # chunks = retrieve(question) 
    # extracted_texts = [chunk["text"] for chunk in chunks]
    
    # Mocked context for now so you can test the graph locally
    extracted_texts = ["Apple plans to be 100% carbon neutral by 2030.", "Recycled materials are used in all MacBooks."]
    
    steps.append("retrieval_node")
    
    return {
        "context": extracted_texts,
        "steps": steps
    }


def grader_node(state: GraphState):
    """
    Node 3: The CRAG addition. Evaluates if the retrieved chunks are actually relevant to the question.
    """
    print("---NODE: GRADER---")
    question = state["question"]
    context_list = state.get("context", [])
    steps = state.get("steps", [])
    
    # Combine the list of retrieved chunks into one big string for the LLM to read
    context_str = "\n\n".join(context_list)
    
    prompt = f"""
    You are a strict grading assistant evaluating the relevance of a retrieved document to a user's question.
    
    Goal: Determine if the retrieved document contains relevant information to answer the question.
    
    Retrieved Document:
    {context_str}
    
    User Question: {question}
    
    Instructions:
    - If the document contains facts, data, or context that helps answer the question, output strictly the word 'relevant'.
    - If the document is completely unrelated, off-topic, or does not contain helpful information, output strictly the word 'irrelevant'.
    - Provide NO explanations, NO preamble, and NO punctuation. Just the single word.
    """

    # We use temperature=0 because we want strict, deterministic grading, not creativity.
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0 
    )
    
    # Clean the output just in case the LLM adds a space or capitalizes it
    raw_relevance = response.choices[0].message.content.strip().lower()
    
    # Failsafe: Sometimes LLMs ignore the "no explanation" rule. 
    # This guarantees we only ever pass "relevant" or "irrelevant" to the routing logic.
    if "irrelevant" in raw_relevance:
        relevance = "irrelevant"
    else:
        relevance = "relevant"

    print(f"---GRADER RESULT: {relevance.upper()}---")
    steps.append("grader_node")
    
    return {
        "relevance": relevance,
        "steps": steps
    }

def generator_node(state: GraphState):
    """
    Node 4: Generates the final answer using the validated context.
    """
    print("---NODE: GENERATOR---")
    question = state["question"]
    context = state.get("context", [])
    steps = state.get("steps", [])
    
    # FUTURE CODE (Integration):
    # answer = generate(question, context)
    
    answer = "This is a mocked final answer generated based on the FAISS context."
    
    steps.append("generator_node")
    
    return {
        "answer": answer,
        "steps": steps
    }