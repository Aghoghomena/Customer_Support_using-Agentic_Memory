

from utils.myclasses import AgentState
from utils.config import llm,GUIDELINES_PATH, MAX_GUIDELINES
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import os

#----- Functions to load and save guidelines from file -----
def load_guidelines() -> str:
    """Load current guidelines from file."""
    if not os.path.exists(GUIDELINES_PATH):
        return ""
    with open(GUIDELINES_PATH, "r") as f:
        return f.read().strip()
    

def save_guidelines(guidelines: str):
    """Save updated guidelines to file."""
    os.makedirs(os.path.dirname(GUIDELINES_PATH), exist_ok=True)
    with open(GUIDELINES_PATH, "w") as f:
        f.write(guidelines)
    print(f"  [Guidelines] Saved updated guidelines")

def format_guidelines_for_prompt() -> str:
    """Returns guidelines ready to inject into agent system prompt."""
    guidelines = load_guidelines()
    if not guidelines:
        return ""
    return f"## Guidelines\n{guidelines}"



# --- Nodes ---
def evaluate_node(state: AgentState) -> AgentState:
    """
    LLM-as-Judge compares agent response to model answer.
    Reflects on what could be improved in tone, steps, or approach.
    """
    print(f"  [Guideline Agent] Judging response quality")

    current_guidelines = load_guidelines()

    response = llm.invoke([
        SystemMessage(content=f"""You are an expert customer service quality judge.
                    Your job is to compare a customer service agent's response to a model answer
                    and update a set of guidelines to help the agent improve future responses.
                    You are maintaining a continuously updated list of the most important procedural behavior instructions for an AI assistant in a customer service setting. These guidelines are based on past interactions and are meant to help the assistant improve over time.

                    CURRENT GUIDELINES:
                    {current_guidelines if current_guidelines else "No guidelines yet."}

                    INSTRUCTIONS:
                    - Compare the agent response to the model answer carefully
                    - Identify differences in: tone, structure, missing information, wrong approach
                    - Update the guidelines list to capture lessons learned
                    - Keep the guidelines concise — maximum {MAX_GUIDELINES} points total
                    - Each guideline should be one clear actionable sentence
                    - You may remove outdated or redundant guidelines to stay within the limit
                    - If the agent response was good, guidelines may not need updating
                    - Return ONLY the updated guidelines list, one per line, no numbering
                    - If no changes needed respond with: NO_CHANGE"""),
                            HumanMessage(content=f"""Category: {state['category']}
                    Intent: {state['intent']}

                    Customer query:
                    {state['query']}

                    Model answer:
                    {state['model_answer']}

                    Agent response:
                    {state['agent_response']}

                    Update the guidelines based on this interaction."""),
    ])

    output = response.content.strip()
    state["updated_guidelines"] = output
    print(f"  [Guideline Agent] Judge output: {output[:120]}...")
    return state


def update_guidelines_node(state: AgentState) -> AgentState:
    """Save updated guidelines to file — no LLM needed."""
    if state["updated_guidelines"] == "NO_CHANGE":
        print(f"  [Guideline Agent] No changes needed")
        return state

    save_guidelines(state["updated_guidelines"])
    return state


# --- Graph ---

def build_guideline_agent():
    builder = StateGraph(AgentState)

    builder.add_node("judge", evaluate_node)
    builder.add_node("update_guidelines", update_guidelines_node)

    builder.set_entry_point("judge")
    builder.add_edge("judge", "update_guidelines")
    builder.add_edge("update_guidelines", END)

    return builder.compile()


guideline_agent = build_guideline_agent()