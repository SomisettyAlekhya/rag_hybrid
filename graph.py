
from langgraph.graph import StateGraph
from agents import agent_reasoning, coding_agent

def create_graph():
    workflow = StateGraph(dict)

    workflow.add_node("reason", agent_reasoning)
    workflow.add_node("coder", coding_agent)

    workflow.set_entry_point("reason")
    workflow.add_edge("reason", "coder")

    return workflow.compile()
