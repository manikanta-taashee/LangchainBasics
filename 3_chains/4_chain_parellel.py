from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

load_dotenv()

client = ChatGroq()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer"),
        ("human", "List the main features of the product {product_name}."),
    ]
)


def analyse_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given these features :{features} , list the pros of the product."),
        ]
    )
    return pros_template.format_prompt(features=features)


def analyse_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given these features :{features} , list the cons of the product."),
        ]
    )
    return cons_template.format_prompt(features=features)


def combine_results(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


pros_branch_chain = (
    RunnableLambda(lambda x: analyse_pros(x)) | client | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyse_cons(x)) | client | StrOutputParser()
)

chain = (
    prompt_template
    | client
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_results(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name": "iPhone 15"})
print(result)
