"""State definition for LangGraph PC Builder Agent."""

from typing import Dict, List, Optional, TypedDict
from src.models.build import PCBuild, ValidationResult
from src.models.user_request import UserRequirements


class AgentState(TypedDict):
    user_query: str
    requirements: Optional[UserRequirements]
    budget_allocation: Optional[Dict[str, int]]
    current_build: Optional[PCBuild]
    validation_result: Optional[ValidationResult]
    iteration_count: int
    feedback_history: List[str]
    final_response: Optional[str]
    parser_source: Optional[str]
