from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState,END
from langgraph.types import Command
from langchain.agents import create_agent
from langgraph.graph import END
from utils.config import llm,SKILL_APPROACH
from utils.myclasses import AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from agents.service_agent import service_agent
from agents.skill_agent import skill_agent
from agents.skill_file_agent import skill_file_agent   # approach 2 — file based
from agents.guideline_agent import guideline_agent

import os







# --- Nodes ---

def supervisor_node(state: AgentState) -> AgentState:
    """
    Supervisor decides what to do next based on current state.
    - No skills_context yet → retrieve skills first
    - No response yet → send to service agent
    - Used model answer → ingest skill
    - Otherwise → done
    """
    print(f"\n[Supervisor] Reviewing state...")

    if state.get("skills_context") is None:
        print(f"  [Supervisor] No skills yet — retrieve first")
        state["next"] = "skill_retrieval"
    elif state.get("response") is None:
        print(f"  [Supervisor] No response yet — send to service agent")
        state["next"] = "service_agent"
    
    # Fix — ingests from every training example that has a model answer
    # elif state["training_mode"] and state.get("model_answer") and not state.get("extracted_skill"):
    #     print(f"  [Supervisor] Agent needed help — delegating ingestion to skill agent")
    #     state["next"] = "skill_ingestion"

    elif state["training_mode"] and state["used_model_answer"] and not state.get("extracted_skill"):
        print(f"  [Supervisor] Agent needed help — delegating ingestion to skill agent")
        state["next"] = "skill_ingestion"
        
    elif state["training_mode"] and state.get("response") and not state.get("guidelines_updated"):
        print(f"  [Supervisor] Step 4 — delegate guideline development")
        state["next"] = "guideline"
    else:
        print(f"  [Supervisor] Response given without model answer — done")
        state["next"] = "end"

    return state

def retrieve_skill_node(state: AgentState) -> AgentState:
    """Supervisor delegated — skill agent retrieves relevant skills."""
    print(f"  [Supervisor] Delegating skill retrieval to skill agent (approach {SKILL_APPROACH})")

    theagent = skill_agent if SKILL_APPROACH == 1 else skill_file_agent

    result = theagent.invoke({
        "query": state["query"],
        "category": state["category"],
        "intent": state["intent"],
        "model_answer": None,
        "agent_response": None,
        "skills_context": "",
        "mode": "retrieve",
    })

    state["skills_context"] = result["skills_context"]
    return state

def service_agent_node(state: AgentState) -> AgentState:
    """Supervisor delegated — service agent handles the query."""
    print(f"  [Service Agent] Handling query")

    result = service_agent.invoke({
        "query": state["query"],
        "model_answer": state.get("model_answer"),
        "skills_context": state["skills_context"],
        "response": None,
        "used_model_answer": False,
        "training_mode": state["training_mode"],
    })

    state["response"] = result["response"]
    state["used_model_answer"] = result["used_model_answer"]
    return state


def skill_ingestion_node(state: AgentState) -> AgentState:
    """Supervisor delegated — skill agent ingests new skill."""
    print(f"  [Supervisor] Delegating skill ingestion to skill agent (approach {SKILL_APPROACH})")

    theagent = skill_agent if SKILL_APPROACH == 1 else skill_file_agent

    result = theagent.invoke({
        "query": state["query"],
        "category": state["category"],
        "intent": state["intent"],
        "model_answer": state["model_answer"],
        "agent_response": state["response"],
        "skills_context": "",
        "extracted_skill": None, 
        "mode": "ingest",
        "extracted_summary": None, 
    })

    state["extracted_skill"] = result.get("agent_response", "ingested")
    return state

def guideline_node(state: AgentState) -> AgentState:
    """Supervisor delegates guideline development to guideline agent."""
    print(f"  [Supervisor] Delegating to guideline agent")

    guideline_agent.invoke({
        "query": state["query"],
        "model_answer": state["model_answer"],
        "agent_response": state["response"],
        "category": state["category"],
        "intent": state["intent"],
        "updated_guidelines": None,
    })

    state["guidelines_updated"] = True
    return state


# --- Routing ---

def route(state: AgentState) -> str:
    return state["next"]


# --- Graph ---
def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("skill_retrieval", retrieve_skill_node)
    builder.add_node("service_agent", service_agent_node)
    builder.add_node("skill_ingestion", skill_ingestion_node)
    builder.add_node("guideline", guideline_node)

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route,
        {
            "skill_retrieval": "skill_retrieval",
            "service_agent": "service_agent",
            "skill_ingestion": "skill_ingestion",
            "guideline": "guideline",
            "end": END,
        }
    )

    # Every agent reports back to supervisor after completing
    builder.add_edge("skill_retrieval", "supervisor")
    builder.add_edge("service_agent", "supervisor")
    builder.add_edge("skill_ingestion", "supervisor")
    builder.add_edge("guideline", "supervisor")

    return builder

    


# Singleton — import this in other modules
mygraph = build_graph().compile()
    # Or save as PNG (requires pygraphviz)
try:
    mygraph.get_graph().draw_mermaid_png(output_file_path="supervisor_graph.png")
    print("\nGraph saved as supervisor_graph.png")
except Exception as e:
    print(f"\nCould not save PNG (pygraphviz may not be installed): {e}")