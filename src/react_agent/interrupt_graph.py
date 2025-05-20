"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from typing import Annotated, Dict, List, Literal, cast, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import get_tools, search
from react_agent.utils import load_chat_model

import time

class MailPayload(BaseModel):
    """Email content that requires approval"""
    subject: str = Field(
        description="Subject of the email that is about to be sent", 
        title="Mail Subject"
    )
    body: str = Field(
        description="Body of the email that is about to be sent", 
        title="Mail Body"
    )
    recipients: List[str] = Field(
        description="List of recipients of the email", 
        title="Mail recipients"
    )

class ResumePayload(BaseModel):
    """User response for email approval"""
    reason: Optional[str] = Field(
        default=None,
        description="Reason to approve or decline", 
        title="Approval Reason"
    )
    approved: bool = Field(
        description="True if approved, False if declined", 
        title="Approval Decision"
    )

class EmailApprovalInterrupt(BaseModel):
    """Interrupt for email approval workflow"""
    interrupt_type: str = Field(
        default="mail_send_approval",
        description="Type of interrupt"
    )
    interrupt_payload: MailPayload = Field(
        description="Description of the email",
        title="Mail Approval Payload"
    )
    resume_payload: ResumePayload = Field(
        description="User Approval for this email",
        title="Email Approval Input"
    )

def another_func():
    return "another_dummy"


# Define the function that calls the model
async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    # answer = interrupt('how are you?')
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # interrupt_schema = [EmailApprovalInterrupt.model_json_schema()]
    # answer = interrupt(interrupt_schema)    
    configuration = Configuration.from_runnable_config(config)
    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools([search])

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )
    # response = AIMessage(
    #     id="1",
    #     content="Hello, world!",
    #     tool_calls=[],
    # )
    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    answer = interrupt('approve this tool call?')
    if answer == 'yes':
        return "tools"
    else:
        return "__end__"


async def make_graph(config: RunnableConfig):
    # Define a new graph
    builder = StateGraph(State, input=InputState, config_schema=Configuration)

    # Define the two nodes we will cycle between
    builder.add_node(call_model)
    builder.add_node("tools", ToolNode([search]))

    # Set the entrypoint as `call_model`
    # This means that this node is the first one called
    builder.add_edge("__start__", "call_model")

    # Add a conditional edge to determine the next step after `call_model`
    builder.add_conditional_edges(
        "call_model",
        # After call_model finishes running, the next node(s) are scheduled
        # based on the output from route_model_output
        route_model_output,
    )

    # Add a normal edge from `tools` to `call_model`
    # This creates a cycle: after using tools, we always return to the model
    builder.add_edge("tools", "call_model")

    # Compile the builder into an executable graph
    # You can customize this by adding interrupt points for state updates
    graph = builder.compile(
        interrupt_before=[],  # Add node names here to update state before they're called
        interrupt_after=[],  # Add node names here to update state after they're called
    )
    graph.name = "ReAct Agent Demo"  # This customizes the name in LangSmith
    return graph
