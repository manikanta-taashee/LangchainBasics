from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

client = ChatGroq()

messages = [
    SystemMessage(content="Solve the following math problem"),
    HumanMessage(content="What is 81 divided by 9?")
]

response = client.invoke(messages)
print("Answer: ", response.content)

# Ai Message
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="what is 10 times 5?")
]

response = client.invoke(messages)
print("Answer: ", response.content)
