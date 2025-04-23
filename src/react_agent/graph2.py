"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# Define the function that calls the model


class Foo(BaseModel):
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="The system prompt to use for the agent's interactions. "
        "This prompt sets the context and behavior for the agent.",
        json_schema_extra={"foo": "bar"},
    )


f = Foo()

graph = (
    StateGraph(dict, config_schema=Foo)
    .add_node("foo", lambda x: x)
    .add_edge(START, "foo")
    .compile()
)

print(graph.config_schema().model_json_schema())
