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

    system = """You are a helpful customer service agent. Answer the customer query clearly and professionally in your own words.
            If you do not have enough information to give a good answer,
            respond with exactly: NEED_MORE_INFO
            Only say NEED_MORE_INFO if you genuinely cannot answer well or not confident."""

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
    if content == "NEED_MORE_INFO":
        print(f"  [Service Agent] Not confident — requesting model answer")
        state["confident"] = False
        state["response"] = None
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
    if state["response"] is None and state["training_mode"] and state.get("model_answer"):
            print(f"  [Service Agent] Not confident — consulting model answer")
            return "answer_with_model"
    return END

def build_service_agent():
    builder = StateGraph(AgentState)

    builder.add_node("attempt_answer", answer_with_agent_node)
    builder.add_node("answer_with_model", answer_with_model_node)

    builder.set_entry_point("attempt_answer")
    builder.add_conditional_edges(
        "attempt_answer",
        should_fetch_model_answer,
        {
            "answer_with_model": "answer_with_model",
            END: END,
        }
    )
    builder.add_edge("answer_with_model", END)

    return builder.compile()

service_agent = build_service_agent()
