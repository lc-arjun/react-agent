"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.amadeus import AmadeusClosestAirport
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


async def repeat(
    text: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Repeat the input text."""
    return text + text


def get_tools(config) -> list[Callable[..., Any]]:
    # return [search]
    configuration = Configuration.from_runnable_config(config)
    tools = []
    for tool in configuration.selected_tools:
        if tool == "search":
            tools.append(search)
        elif tool == "repeat":
            tools.append(repeat)
    return tools
