import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings



# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
query = "An intresting fact about programming"

db = Chroma(
    embedding_function=embeddings,
    persist_directory=persistent_directory
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

relevant_docs = retriever.invoke(query)
# print(relevant_docs)
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")