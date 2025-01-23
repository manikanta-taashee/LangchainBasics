import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
query = "An intresting fact about cars"
db = Chroma(
    embedding_function=embeddings,
    persist_directory=persistent_directory
)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
relevant_docs = retriever.invoke(query)
# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")   
combined_input =(
    "Here are some facts about programming:\n" +
    "\n".join([doc.page_content for doc in relevant_docs]) +
    "\n\n" +
    "Answer the question based on the context provided. If the context does not contain the answer, say 'I don't know'" +
    "\n\n" +
    "Question: " + query
)
print(combined_input)
client = ChatGroq(
    model="llama3-8b-8192",
)
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage(content=combined_input)
]
response = client.invoke(messages)
print("______LLM Response______")
print(response.content)
