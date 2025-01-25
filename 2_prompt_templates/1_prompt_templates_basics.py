from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# # PART 1: Create a ChatPromptTemplate using a template string
# template = "Tell me a joke about {topic}"
# prompt_template = ChatPromptTemplate.from_template(template)

# print("==== Prompt From Template ====")

# prompt = prompt_template.invoke({"topic":"AI"})
# print(prompt)

# # Part 2: Prompt with multiple variables

# template_multiple = """ You are a helpful assistant.
# Human: Tell me a {adjective} joke about a {animal}.
# Assistant: """

# prompt_template_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_template_multiple.invoke({"adjective":"funny", "animal":"cat"})
# print("\n----- Prompt with Multiple Placeholders -----\n")
# print(prompt)

# Part 3 : Prompt with system and human messages (using Tuples)

# messages = [
#     ("system","You are a comedian who tells jokes about {topic}"),
#     ("human","Tell me {joke_count} jokes")
# ]

# This will not work
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me {joke_count} jokes."),
# ]


# this will work
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me {joke_count} jokes."),
]

prompt_template_tuple = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template_tuple.invoke({"topic": "AI", "joke_count": 3})
print(prompt)
