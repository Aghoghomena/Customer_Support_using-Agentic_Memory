from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from utils.config import llm, GUIDELINES_PATH
from utils.myclasses import AgentState
import os


def load_guidelines() -> str:
    if not os.path.exists(GUIDELINES_PATH):
        return ""
    with open(GUIDELINES_PATH, "r") as f:
        content = f.read().strip()
    return content

def answer_with_agent_node(state: AgentState) -> AgentState:
    """Agent tries to answer from own knowledge, skills and guidelines."""

    guidelines = load_guidelines()

    system = """You are a customer service agent for a specific company.
        You can ONLY answer using information you have been explicitly provided in your skills and guidelines.

        You do NOT have access to:
        - Company specific policies, procedures or systems
        - Specific portal names, URLs, phone numbers or email addresses
        - Order management systems or customer account details
        - Internal company processes or department names

        If the query requires any company specific information that is NOT in your skills or guidelines,
        respond with exactly: NEED_MORE_INFO

        Only answer confidently if your skills and guidelines contain enough specific information to help.
        if you don't have enough information to give a helpful answer, it's better to say NEED_MORE_INFO than to give a vague or incorrect response.

        if skill and guidelines are provided, use them to try to answer. If they are not sufficient, say NEED_MORE_INFO.
        - Follow the approach and tone described in your skills exactly
        - If your skills say to ask for information first — ask for it, do not give steps directly
        - If your skills say to provide steps — provide them clearly
        - Mirror the structure and tone of your skills as closely as possible
        - Do NOT add information not present in your skills or guidelines
        - Do NOT invent URLs, email addresses, phone numbers or company specific details
        - Do NOT make assumptions about what systems or processes the company uses
        """

    if state.get("skills_context"):
        system += f"\n\n## Relevant skills\n{state['skills_context']}"

    if guidelines:
        system += f"\n\n## Guidelines\n{guidelines}"

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Customer query: {state['query']}"),
    ])

    content = response.content.strip()
    print(f"  [Service Agent] First attempt to answer query: {content[:80]}...")
    if "NEED_MORE_INFO" in content.upper().replace(" ", "_"):
        print(f"  [Service Agent] Not confident — requesting model answer")
        state["confident"] = False
        state["response"] = None
    else:
        state["confident"] = True
        state["response"] = content   # ← this was missing
    return state

def answer_with_model_node(state: AgentState) -> AgentState:
    """Agent answers using the model answer as reference, it must respond."""

    guidelines = load_guidelines()

    system = """You are a helpful customer service agent.
                You have been provided a reference answer below to help you respond.
                You MUST use it to give a full, clear response in your own words.
                Do NOT say NEED_MORE_INFO — you have everything you need."""

    if state.get("skills_context"):
        system += f"\n\n## Relevant skills\n{state['skills_context']}"

    if guidelines:
        system += f"\n\n## Guidelines\n{guidelines}"

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""Customer query: {state['query']}
                    Reference answer (use this to guide your response, do not copy verbatim):
                    {state['model_answer']}"""),
        ])

    content = response.content.strip()
    print(f"  [Service Agent] answer using model as a reference: {content[:80]}...")

    state["response"] = content
    state["used_model_answer"] = True
    state["confident"] = True
    return state


#--- Conditional edge function ---
def should_fetch_model_answer(state: AgentState) -> str:
    """
    Conditional edge after service_agent.
    If not confident AND in training mode and haven't tried model answer yet fetch model answer and retry.
    Already used model answer (second attempt) → END no matter what
    Otherwise → end have to prevent infinite loop.
    """
    if state["response"] is None:
        if state["training_mode"] and state.get("model_answer"):
            print(f"  [Service Agent] Not confident — consulting model answer")
            return "answer_with_model"
        else:
            print(f"  [Service Agent] Not confident — giving best effort answer")
            return "answer_with_no_model"
    return END


def answer_without_fallback(state: AgentState) -> AgentState:
    """
    Only runs in test mode when agent was not confident.
    Forces the agent to give its best answer with no fallback.
    """
    guidelines = load_guidelines()

    system = """You are a helpful customer service agent.
            You must give your best answer to the customer query.
            Use any knowledge you have to provide a helpful response.
            Never refuse to answer or say you don't know — always try your best.
            Yoou have access to the following relevant skills and guidelines to help you answer:"""

    if state.get("skills_context"):
        system += f"\n\n## Relevant skills\n{state['skills_context']}"

    if guidelines:
        system += f"\n\n## Guidelines\n{guidelines}"

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Customer query: {state['query']}"),
    ])

    content = response.content.strip()
    print(f"  [Service Agent] Best effort answer: {content[:80]}...")
    state["response"] = content
    return state

def build_service_agent():
    builder = StateGraph(AgentState)

    builder.add_node("attempt_answer", answer_with_agent_node)
    builder.add_node("answer_with_model", answer_with_model_node)
    builder.add_node("answer_with_no_model", answer_without_fallback)

    builder.set_entry_point("attempt_answer")
    builder.add_conditional_edges(
        "attempt_answer",
        should_fetch_model_answer,
        {
            "answer_with_model": "answer_with_model",
            "answer_with_no_model": "answer_with_no_model",
            END: END,
        }
    )
    builder.add_edge("answer_with_model", END)
    builder.add_edge("answer_with_no_model", END)

    return builder.compile()

service_agent = build_service_agent()
    # Or save as PNG (requires pygraphviz)
try:
    service_agent.get_graph().draw_mermaid_png(output_file_path="service_agent.png")
    print("\nGraph saved as service_agent.png")
except Exception as e:
    print(f"\nCould not save PNG (pygraphviz may not be installed): {e}")
