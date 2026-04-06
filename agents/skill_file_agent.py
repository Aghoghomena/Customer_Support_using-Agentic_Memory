
from agents.skills_file_functions import (get_skills_summary_text,save_skill_from_interaction, get_skill_detail)
from utils.myclasses import AgentState
from utils.config import llm
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
# --- Tools ---
def fetch_skill_detail(skill_id: str) -> str:
        """
        Fetch the full detail of a skill by its skill_id.
        Use the skills summary list to identify the right skill_id first.
        skill_id: the ID of the skill to fetch e.g. skill_001
        """
        detail = get_skill_detail(skill_id)
        return detail

def list_available_skills() -> str:
        """
        List all available skills with their names and descriptions.
        Call this to see what skills are available before fetching details.
        """
        return get_skills_summary_text()



# --- Nodes ---

def retrieve_node(state: AgentState) -> AgentState:
    """
    LLM sees full skills summary list in system prompt,
    selects the most relevant skill, then fetches full detail from file.
    """
    print(f"  [Skill File Agent] Selecting skill from summary list")

    skills_summary = get_skills_summary_text()

    if "No skills available yet" in skills_summary:
        print(f"  [Skill File Agent] No skills available yet")
        state["skills_context"] = ""
        return state
    
    response = llm.invoke([
        SystemMessage(content=f"""You are a skill selection agent.
                You have access to the following customer service skills:

                {skills_summary}

                INSTRUCTIONS:
                - Review the customer query carefully
                - Select the most relevant skill from the list above
                - Respond with ONLY the exact skill name as shown e.g. cancel-order
                - If no skill is relevant respond with exactly: NO_RELEVANT_SKILL"""),
                HumanMessage(content=f"Customer query: {state['query']}"),
    ])

    skill_name = response.content.strip()
    print(f"  [Skill File Agent] Selected: {skill_name}")

    if skill_name == "NO_RELEVANT_SKILL":
        print(f"  [Skill File Agent] No relevant skill found")
        state["skills_context"] = ""
    else:
        # Direct file read — no LLM needed here
        detail = get_skill_detail(skill_name)
        if "not found" in detail.lower():
            print(f"  [Skill File Agent] Skill '{skill_name}' not found in files")
            state["skills_context"] = ""
        else:
            print(f"  [Skill File Agent] Fetched detail: {detail[:80]}...")
            state["skills_context"] = detail

    return state


def extract_node(state: AgentState) -> AgentState:
    """LLM extracts a summary and full skill detail from the interaction."""
    print(f"  [Skill File Agent] Extracting skill with LLM")

    response = llm.invoke([
        SystemMessage(content="""You are a skill extraction agent.
        Extract a reusable skill from a customer service interaction.
        Return your response in this exact format:

        SUMMARY: <one sentence describing what this skill handles>
        DETAIL:
        <3-6 bullet points covering: tone, steps, key information to include, category>

        Do NOT include specific personal details like order numbers or names.
        Keep it general and reusable."""),
        HumanMessage(content=f"""Extract a skill from this interaction:

        Category: {state['category']}
        Intent: {state['intent']}
        Customer query: {state['query']}
        Model answer: {state['model_answer']}
        Agent response: {state['agent_response']}"""),
    ])

    output = response.content.strip()
    print(f"  [Skill File Agent] Extracted: {output[:120]}...")

    # Parse summary and detail
    if "SUMMARY:" in output and "DETAIL:" in output:
        parts = output.split("DETAIL:")
        summary = parts[0].replace("SUMMARY:", "").strip()
        detail = parts[1].strip()
    else:
        summary = f"Handle {state['intent']} queries"
        detail = output

    state["extracted_summary"] = summary
    state["extracted_skill"] = detail
    return state
    


def save_node(state: AgentState) -> AgentState:
    """Direct file save — no LLM needed."""
    print(f"  [Skill File Agent] Saving skill to files")

    save_skill_from_interaction(
        description=state["extracted_summary"],
        content=state["extracted_skill"],
        category=state["category"],
        intent=state["intent"],
        query=state["query"],
    )
    return state

# --- Routing ---
def route_mode(state: AgentState) -> str:
    return state["mode"]

# --- Graph ---

def build_skill_file_agent():
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


skill_file_agent = build_skill_file_agent()