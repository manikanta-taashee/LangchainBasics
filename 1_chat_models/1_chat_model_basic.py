from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

import os
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set in the environment variables")

# groq_api_key = os.getenv("GROQ_API_KEY")


client = ChatGroq(temperature=0.9)

# response = client.invoke("What is the capital of France?")
# print(response.content)

# reponse = client.invoke("Write a 100 words short story about a cat")
# print(reponse.content)

messages = [
    SystemMessage(content="You are a expert blog writer."),
    HumanMessage(content="Write a 100 words blog on Generative AI")
]

response = client.invoke(messages)
print(response.content)
