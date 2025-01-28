from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent , create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables from .env file
load_dotenv()
search = DuckDuckGoSearchRun()


def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    """Searches Wikipedia for the given query."""
    from wikipedia import summary
    try:
        return summary(query, sentences=2)
    except Exception as e:
        return f"Error searching Wikipedia: {e}"

# Define the tools that the agent can use
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to know information about a topic.",
    ),
    
]

prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(
    model="gpt-4o",
)


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    # stop_sequences=True
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

response = agent_executor.invoke({"input": "What is the weather in bengaluru?"})
print(response)