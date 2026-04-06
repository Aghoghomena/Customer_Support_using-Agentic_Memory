from agents.skill_function import retrieve_skills, add_skill
from utils.myclasses import AgentState
from utils.config import llm, TOP_K_SKILLS
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage


SKILL_AGENT_PROMPT = """You are a skill extraction agent.
Your job is to extract a concise, reusable skill from a customer service example.
A skill describes HOW to handle a type of query — the steps, tone, and key information to include.
Write the skill as clear instructions (3-6 bullet points max).
Do NOT include specific order numbers, names, or personal details — keep it general and reusable."""


def retrieve_node(state: AgentState) -> AgentState:
    skills = retrieve_skills(state["query"], top_k=TOP_K_SKILLS)
    state["skills_context"] = skills if skills else ""
    if skills:
        print(f"  [Skill Agent] Found skills: {skills[:80]}...")
    else:
        print(f"  [Skill Agent] No relevant skills found.")
    return state


def extract_node(state: AgentState) -> AgentState:
    """LLM extracts a reusable skill from the interaction."""
    print(f"  [Skill Agent] Extracting skill with LLM")

    response = llm.invoke([
        SystemMessage(content=SKILL_AGENT_PROMPT),
        HumanMessage(content=f"""Extract a reusable skill from this interaction:

Category: {state['category']}
Intent: {state['intent']}
Customer query: {state['query']}
Model answer: {state['model_answer']}
Agent response: {state['agent_response']}"""),
    ])

    extracted = response.content.strip()
    print(f"  [Skill Agent] Extracted skill: {extracted[:120]}...")
    state["extracted_skill"] = extracted
    return state


def save_node(state: AgentState) -> AgentState:
    """Saves the extracted skill to ChromaDB."""
    print(f"  [Skill Agent] Saving skill to ChromaDB")
    add_skill(
        query=state["query"],
        skill_text=state["extracted_skill"],
        category=state["category"],
        intent=state["intent"],
    )
    return state


# --- Routing ---

def route_mode(state: AgentState) -> str:
    return state["mode"]


# --- Build Skill Agent Subgraph ---

def build_skill_agent():
    builder = StateGraph(AgentState)
    builder.add_node("router", lambda state: state)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("extract", extract_node)
    builder.add_node("save", save_node)

    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        route_mode,
        {
            "retrieve": "retrieve",
            "ingest": "extract",
        }
    )

    builder.add_edge("retrieve", END)
    builder.add_edge("extract", "save")
    builder.add_edge("save", END)
    return builder.compile()


skill_agent = build_skill_agent()
