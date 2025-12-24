# model-context-protocol
This repo consists of a basic agent structure by using tool from Bright Data MCP server. 

In the stock recommender code I was facing this below mentioned error even after trying lots of things. But finally Gemini 3.0 gave me the solution. 

The error NotImplementedError: StructuredTool does not support sync invocation occurs because you are using Model Context Protocol (MCP) tools (which are inherently asynchronous) inside a LangGraph workflow that is attempting to execute them synchronously.

Even though your run_agent function is async, the internal supervisor.stream loop and the way the agents are compiled are defaulting to synchronous execution for the tool calls.

The Solution: Await the Stream
To fix this, you need to change how you iterate through the supervisor's results. LangGraph provides an async streaming method (astream) for exactly this scenario.

Here are the specific changes you need to make:

1. Use astream instead of stream
Replace your synchronous for loop with an async for loop.

Change this:

Python

for chunk in supervisor.stream(
    {"messages": [{"role": "user", "content": query}]},
):
    pretty_print_messages(chunk, last_message=True)
To this:

Python

async for chunk in supervisor.astream(
    {"messages": [{"role": "user", "content": query}]},
    # subgraphs=True # Optional: add if you want to see inner agent steps
):
    pretty_print_messages(chunk, last_message=True)