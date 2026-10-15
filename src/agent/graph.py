"""PC Builder LangGraph workflow assembly."""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    parse_requirements_node,
    plan_and_select_node,
    validate_build_node,
    should_continue,
    explain_build_node,
)


def create_pc_builder_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # 1. Register nodes
    workflow.add_node("parse_requirements", parse_requirements_node)
    workflow.add_node("plan_and_select", plan_and_select_node)
    workflow.add_node("validate_build", validate_build_node)
    workflow.add_node("explain_build", explain_build_node)

    # 2. Define edges
    workflow.set_entry_point("parse_requirements")
    workflow.add_edge("parse_requirements", "plan_and_select")
    workflow.add_edge("plan_and_select", "validate_build")

    # 3. Conditional routing with self-correction loop
    workflow.add_conditional_edges(
        "validate_build",
        should_continue,
        {
            "retry": "plan_and_select",
            "explain": "explain_build",
        },
    )

    workflow.add_edge("explain_build", END)

    return workflow.compile()


app = create_pc_builder_graph()
