from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os

from langchain_core.tools import Tool
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

from dotenv import load_dotenv
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor,create_openai_functions_agent,create_tool_calling_agent
from tools.sql import run_query_tool
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant that can run SQL queries"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

tools = [run_query_tool]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

agent_executor.invoke({"input": "How many open orders are there?"})