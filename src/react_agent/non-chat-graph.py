from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class State(BaseModel):
    foo: str

def dummy_function(state: State):
    return {"foo": "bar"}


def dummy_function2(state: State):
    return {"foo": "baz"}


workflow = StateGraph(State)
workflow.add_node("dummy_function", dummy_function)
workflow.add_node("dummy_function2", dummy_function2)

workflow.add_edge(START, "dummy_function")
workflow.add_edge("dummy_function", "dummy_function2")
workflow.add_edge("dummy_function2", END)

graph = workflow.compile()