
from typing import TypedDict, Optional


class AgentState(TypedDict):
    query: str
    category: str
    intent: str
    model_answer: Optional[str]
    skills_context: str
    response: Optional[str]
    used_model_answer: bool
    training_mode: bool
    confident: bool
    next: str              # ← add this
    agent_response: Optional[str]
    extracted_skill: Optional[str]
    mode: str 
    skill_mode:str
    extracted_summary: Optional[str]
    extracted_skill: Optional[str]
    updated_guidelines: Optional[str]
    guidelines_updated: bool 



class SkillAgentState(TypedDict):
    query: str
    category: str
    intent: str
    model_answer: Optional[str]
    agent_response: Optional[str]
    skills_context: str
    mode: str 


class SkillFileAgentState(TypedDict):
    query: str
    category: str
    intent: str
    model_answer: Optional[str]
    agent_response: Optional[str]
    skills_context: str             # full detail of selected skill
    extracted_skill: Optional[str]
    extracted_summary: Optional[str]
    mode: str 