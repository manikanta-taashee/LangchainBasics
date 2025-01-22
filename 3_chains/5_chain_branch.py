from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

load_dotenv()

client = ChatGroq()

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Classify the sentiment of the following text as positive, negative or neutral: {feedback}"),
    ]
)


positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a thank you note for this positive feedback : {feedback}"),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a response addressing the concerns of this negative feedback: {feedback}"),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a response for this neutral feedback: {feedback}"),
    ]
)

brances = RunnableBranch(
    (lambda x: "positive" in x.lower(),
     positive_feedback_template | client | StrOutputParser()
    ),
    (lambda x: "negative" in x.lower(),
     negative_feedback_template | client | StrOutputParser()
    ),
    (lambda x: "neutral" in x.lower(),
     neutral_feedback_template | client | StrOutputParser()
    ),
    # Default case when no condition matches
    neutral_feedback_template | client | StrOutputParser()
)
# Create a classification chain
classification_chain = classification_template | client | StrOutputParser()

# Create a chain that uses the classification chain to determine which branch to use
chain = classification_chain | brances

review = "The product is excellent. I really enjoyed using it and found it very helpful."
result = chain.invoke({"feedback": review})
print(result)