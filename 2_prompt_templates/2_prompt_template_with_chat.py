from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

client = ChatGroq()

# Part 1: Create a chatprompt template using a template string

# template = "Tell me a joke about {topic}"
# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({"topic":"AI"})

# print("Prompt: ", prompt)
# result = client.invoke(prompt)
# print("Result: ", result.content)

# part 2 : with multiple variables

# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} short story about a {animal}.
# Assistant:"""
# prompt_template_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_template_multiple.invoke({"adjective":"funny", "animal":"cat"})
# print("Prompt: ", prompt)
# result = client.invoke(prompt)
# print("Result: ", result.content)

# Part 3 : Prompt with system and human messages (using Tuples)

# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}"),
#     ("human", "Tell me {joke_count} jokes")
# ]

# prompt_template_tuple = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template_tuple.invoke({"topic":"AI", "joke_count":3})
# print(prompt)

# response = client.invoke(prompt)
# print(response.content)

messages = [
    ("system", "You are an expert article writer who writes articles about {topic}"),
    ("human", "Write {article_count} articles about {topic}")
]

prompt_template_messages = ChatPromptTemplate.from_messages(messages)
# take topic as user input
user_topic = input("Enter the topic: ")
user_count = input("Enter the number of articles: ")
prompt = prompt_template_messages.invoke({"topic": user_topic, "article_count": user_count})
# print(prompt)

response = client.invoke(prompt)
print(response.content)
