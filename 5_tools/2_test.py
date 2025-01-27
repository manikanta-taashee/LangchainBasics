from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os

from langchain_core.tools import Tool
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from dotenv import load_dotenv
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor,create_openai_functions_agent,create_tool_calling_agent

load_dotenv()


def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p") 

def get_current_file_path(*args):
    """
    Returns the absolute path of the current file.
    """
    return os.path.abspath(__file__)



tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful when you need to know the current time"
    ),
    Tool(
        name="where_am_i",
        func=get_current_file_path,
        description="Useful when you need to know the current file path"
    )
]
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question with your tools"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# Initialize the Language Model using Groq
# temperature=0.7 provides a good balance between creativity and accuracy
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0.7,
# )

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

agent_executor.invoke({"input": "Where this file 2_test.py is located?"})