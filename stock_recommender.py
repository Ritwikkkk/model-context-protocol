import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_supervisor import create_supervisor
from langchain.agents import create_agent
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("mcp").setLevel(logging.CRITICAL)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.CRITICAL)
logging.getLogger("langchain.tools").setLevel(logging.CRITICAL)
logging.disable(logging.CRITICAL)


load_dotenv()


from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


async def run_agent(query):

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

    stock_finder_agent = create_agent(model=llm, tools=tools, 
            system_prompt="""You are a stock research analyst specializing in the Indian Stock Market - NSE. 
            Your task is to select 2 promising, actively traded NSE-listed stocks for short term trading (buy/sell) 
            based on recent performance, news buzz, volume or technical strength.
            Avoid penny stocks and illiquid companies.
            Output should include stock names, tickers and brief reasoning for each choice.
            Respond in structured plain text format.""", name="stock_finder_agent")

    market_data_agent = create_agent(model=llm, tools=tools, 
            system_prompt="""You are a market data analyst for Indian stocks listed on NSE. Given a list of stock tickers (eg RELIANCE, INFY), your task is to gather recent market information for each stock, including:
    - Current price
    - Previous closing price
    - Today's volume
    - 7-day and 30-day price trend
    - Basic Technical indicators (RSI, 50/200-day moving averages)
    - Any notable spkies in volume or volatility
    
    Return your findings in a structured and readable format for each stock, suitable for further analysis by a recommendation engine. Use INR as the currency. Be concise but complete."""
    , name = "market_data_agent")

    news_analyst_agent = create_agent(model=llm, tools=tools, 
            system_prompt="""You are a financial news analyst. Given the names or the tickers of Indian NSE listed stocks, your job is to-
    - Search for the most recent news articles (past 3-5 days)
    - Summarize key updates, announcements, and events for each stock
    - Classify each piece of news as positive, negative or neutral
    - Highlist how the news might affect short term stock price
                                            
    Present your response in a clear, structured format - one section per stock.

    Use bullet points where necessary. Keep it short, factual and analysis-oriented""", 
    name = "news_analyst_agent")

    price_recommender_agent = create_agent(model=llm, tools=tools, 
            system_prompt="""You are a trading stratefy advisor for the Indian Stock Market. You are given -
    - Recent market data (current price, volume, trend, indicators)
    - News summaries and sentiment for each stock
        
    Based on this info, for each stock-
    1. Recommend an action : Buy, Sell or Hold
    2. Suggest a specific target price for entry or exit (INR)
    3. Briefly explain the reason behind your recommendation.
        
    Your goal is to provide practical. near-term trading advice for the next trading day.
        
    Keep the response concise and clearly structured.""", name = "price_recommender_agent")


    supervisor = create_supervisor(
    model=llm,
    agents=[stock_finder_agent, market_data_agent, news_analyst_agent, price_recommender_agent],
    prompt=(
        "You are an orchestrator. Follow this EXACT sequence:\n"
        "1. Ask stock_finder_agent to find 2 NSE stocks.\n"
        "2. Once you have the tickers, ask market_data_agent to get prices for those SPECIFIC tickers.\n"
        "3. Then ask news_analyst_agent for news on those tickers.\n"
        "4. Finally, ask price_recommender_agent for the final Buy/Sell advice.\n"
        "If an agent returns without data, tell them to try again using their tools."
    ),
    add_handoff_back_messages=True).compile()

    # for chunk in supervisor.stream(
    # {
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": query,
    #             }
    #         ]
    #     },
    # ):
    #     pretty_print_messages(chunk, last_message=True)
    async for chunk in supervisor.astream(
        {"messages": [{"role": "user", "content": query}]},
        # subgraphs=True # Optional: add if you want to see inner agent steps
    ):
        pretty_print_messages(chunk, last_message=True)

    final_message_history = chunk["supervisor"]["messages"]


if __name__ == "__main__":
    asyncio.run(run_agent("Give me good stock recommendation from NSE"))
