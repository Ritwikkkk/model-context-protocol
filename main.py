import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("mcp").setLevel(logging.CRITICAL)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.CRITICAL)
logging.getLogger("langchain.tools").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)

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
    system_prompt="You are a web search agent with access to brightdata tool to get latest data.",
    )
    agent_response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "How many runs did Rohit Sharma score in today's Vijay Hazare Trophy match against AP?"}]
    })
    last_message = agent_response["messages"][-1]

    # If the content is a list, extract the text from the first item
    if isinstance(last_message.content, list):
        text_output = last_message.content[0].get('text', '')
    else:
        text_output = last_message.content

    print(text_output)

if __name__ == "__main__":
    asyncio.run(run_agent())
