from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

client = ChatGroq()

# use this to store the chat history
chat_history = []

system_message = SystemMessage(content="You are a helpful assistant")
chat_history.append(system_message)

while True:
    print()
    query = input("Enter your query: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = client.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    print("AI: ", response)

print("--- Meeting Ended ---")
print("Chat History: ", chat_history)
