"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Annotated, Literal, Optional

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """The configuration for the agent."""

    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="The system prompt to use for the agent's interactions. "
        "This prompt sets the context and behavior for the agent.",
        json_schema_extra={"langgraph_nodes": ["call_model"], "langgraph_type": "prompt"},
    )

    model: Annotated[
            Literal[
                "anthropic/claude-3-7-sonnet-latest",
                "anthropic/claude-3-5-haiku-latest",
                "openai/o1",
                "openai/gpt-4o-mini",
                "openai/o1-mini",
                "openai/o3-mini",
            ],
            {"__template_metadata__": {"kind": "llm"}},
        ] = Field(
            default="openai/gpt-4o-mini",
            description="The name of the language model to use for the agent's main interactions. "
        "Should be in the form: provider/model-name.",
        json_schema_extra={"langgraph_nodes": ["call_model"]},
    )

    selected_tools: list[Literal["search", "repeat"]] = Field(
        default_factory=list,
        description="The list of tools to use for the agent's interactions. "
        "This list should contain the names of the tools to use.",
        json_schema_extra={"langgraph_nodes": ["tools"]},
    )

    max_search_results: int = Field(
        default=10,
        description="The maximum number of search results to return for each search query."
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config_dict = ensure_config(config)
        configurable = config_dict.get("configurable") or {}
        # Use model_fields instead of fields() in Pydantic v2
        return cls(**{k: v for k, v in configurable.items() if k in cls.model_fields})
