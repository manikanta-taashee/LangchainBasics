from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

client = ChatGroq()


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes")
    ]
)

# chain = prompt_template | client
chain = prompt_template | client | StrOutputParser() 

result = chain.invoke({"topic":"Doctors", "joke_count":3})
print(result)