import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chromadb_books")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def query_vector_store(store_name,query,embedding_function,search_type,search_kwargs):
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")
        
# Define the user's question
query = "How did Juliet die?"



#1 Similarity Search
# this method retrieves the documents based on vector similarity
# It used cosine similarity to query the vector store
# 
print("------Using Similarity Search------")
query_vector_store("chromadb_books",query,embeddings,"similarity",{"k": 3})

# 2. MMR max marginal Relevance
print("------Using MMR------")
# MMR is a method that retrieves the documents based on vector similarity
# It used cosine similarity to query the vector store
# It is a greedy algorithm that maximizes the relevance of the retrieved documents
# It is a good method for retrieving the most relevant documents
query_vector_store("chromadb_books",query,embeddings,"mmr",{"k": 3, "fetch_k": 20, "lambda_mult": 0.5})

# 3. Similarity Search Threshold
print("------Using Similarity Search Threshold------")
query_vector_store("chromadb_books",query,embeddings,"similarity_score_threshold",{"k": 3, "threshold": 0.1})

