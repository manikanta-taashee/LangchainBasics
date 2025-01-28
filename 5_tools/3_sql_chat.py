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
from langchain.agents import  AgentExecutor,create_tool_calling_agent
from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool

load_dotenv()

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
# )


# Initialize the Language Model using Groq
# temperature=0.7 provides a good balance between creativity and accuracy
# llm = ChatGroq(
#     model="llama3-70b-8192",
#     temperature=0.7,
#     # model="mixtral-8x7b-32768"
# )
llm = ChatOpenAI(
    model="gpt-4o-mini",
    # temperature=0,
)

tables = list_tables()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI that has access to a SQLite database.\n"
            f"The database has tables of: {tables}\n"
            "Do no make any assumptions about what tables exist or what columns exist.\n"
            "If you need to describe a table, use the describe_tables tool.\n"
            "If you need to run a SQL query, use the run_sqlite_query tool.\n"
        )),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)
tools = [run_query_tool, describe_tables_tool, write_report_tool]

print(prompt)

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# agent_executor.invoke({"input": "Create a new user called 'John Doe' with email 'john.doe@example.com' and password 'password123'"})
agent_executor.invoke({"input": "How many open orders are there"})
