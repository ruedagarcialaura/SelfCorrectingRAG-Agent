from langgraph.graph import StateGraph, END
from state import GraphState
from agents import router_node, retrieval_node, grader_node, generator_node

# ==========================================
# 1. DEFINE THE CONDITIONAL ROUTING LOGIC
# ==========================================

def decide_route_after_router(state: GraphState):
    """
    Reads the classification from the Router Agent and decides the next step.
    """
    classification = state.get("classification")
    
    if classification == "apple_quest":
        print("---ROUTER DECISION: Proceed to FAISS Retrieval---")
        return "retrieve"
    else:
        print(f"---ROUTER DECISION: '{classification}'. Skipping FAISS.---")
        return "generate"

def decide_route_after_grader(state: GraphState):
    """
    Reads the relevance from the Grader Agent and decides what to do next.
    """
    relevance = state.get("relevance")
    
    if relevance == "relevant":
        print("---GRADER DECISION: Context is useful. Proceed to Generation.---")
        return "generate"
    else:
        print("---GRADER DECISION: Context is irrelevant. (Fallback triggered)---")
        # In a full CRAG, you would route to a "web_search" node here.
        # For now, we route to generation, but the generator will know the context failed.
        return "generate"


# ==========================================
# 2. BUILD THE GRAPH (The Workflow)
# ==========================================

# Initialize the graph with your custom state dictionary
workflow = StateGraph(GraphState)

# Add all the nodes (functions) you created in agents.py
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("grader", grader_node)
workflow.add_node("generate", generator_node)

# Set the entry point (where the workflow always starts)
workflow.set_entry_point("router")

# ==========================================
# 3. CONNECT THE EDGES (The Paths)
# ==========================================

# Path leaving the Router (Conditional)
workflow.add_conditional_edges(
    "router",
    decide_route_after_router,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

# Path leaving Retrieval (Strict straight line to the Grader)
workflow.add_edge("retrieve", "grader")

# Path leaving the Grader (Conditional)
workflow.add_conditional_edges(
    "grader",
    decide_route_after_grader,
    {
        "generate": "generate"
        # "web_search": "web_search" # You can add this node later!
    }
)

# Path leaving the Generator (Ends the workflow)
workflow.add_edge("generate", END)

# ==========================================
# 4. COMPILE THE APP
# ==========================================
app = workflow.compile()


# ==========================================
# 5. TEST RUNNER
# ==========================================
if __name__ == "__main__":
    # Test 1: A valid Apple question
    print("\n\n" + "="*50)
    print("TEST 1: Valid Apple Question")
    print("="*50)
    inputs_1 = {"question": "What is Apple's renewable energy goal?", "steps": []}
    
    for output in app.stream(inputs_1):
        for key, value in output.items():
            pass # The print statements inside the nodes will show the progress
            
    print(f"\nFinal State Steps: {value.get('steps')}")


    # Test 2: An out-of-scope question
    print("\n\n" + "="*50)
    print("TEST 2: Out of Scope Question")
    print("="*50)
    inputs_2 = {"question": "What is the recipe for a classic lasagna?", "steps": []}
    
    for output in app.stream(inputs_2):
        for key, value in output.items():
            pass 
            
    print(f"\nFinal State Steps: {value.get('steps')}")
