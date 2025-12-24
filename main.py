import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv()

async def run_agent():

    client = MultiServerMCPClient(
        {
            "bright_data": {
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.getenv("BRIGHT_DATA_API_KEY"),
                },
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    )
    agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a web search agent with access to brightdata tool to get latest data."
    )
    agent_response = await agent.ainvoke({"messages": "How many runs did Virat Kohli score in today's Vijay Hazare Trophy match against AP?"})
    print(agent_response["messages"][-1].content)
    
if __name__ == "main":
    asyncio.run(run_agent())
