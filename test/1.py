from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from pydantic import BaseModel  # Add this import at the top
# Load environment variables from .env file
load_dotenv()


# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format

def write_to_text_file(text: str) -> str:
    """Write the given text to time.txt file."""
    try:
        with open('time.txt', 'w') as file:
            file.write(text)
        return f"Successfully wrote '{text}' to time.txt"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

# List of tools available to the agent
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time",
    ),
    Tool(
        name="Write to Text File",
        func=write_to_text_file,
        description="Writes the provided text to a file named 'time.txt'. Just provide the text to write.",
    ),
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

# Initialize a ChatOpenAI model
# llm = ChatOpenAI(
#     model="gpt-4o", temperature=0
# )

llm = ChatGroq(
    # model="llama3-70b-8192",
    model="mixtral-8x7b-32768",
    # model="llama3-8b-8192",
    temperature=0.7,
)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Run the agent with a test query
response = agent_executor.invoke({"input": "What time is it? and write the time to a text file called 'time.txt'"})

# Print the response from the agent
print("response:", response)
 