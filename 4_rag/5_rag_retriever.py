import os
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch Pinecone API key from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Check if the Pinecone API key was loaded correctly
if not pinecone_api_key:
    raise ValueError("Pinecone API key not found. Make sure you have set it in the .env file.")

# Initialize Pinecone with your API key
pc = Pinecone(api_key=pinecone_api_key)

# Define Pinecone index name and namespace
index_name = "llm-training-facts"
namespace = ""  # Empty string means no namespace; adjust if needed

# Initialize the Pinecone index
index = pc.Index(index_name)

# Define the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define the query
query = "An interesting fact about programming"

# Create the embedding for the query
query_embedding = embeddings.embed_query(query)

# Perform a similarity search in Pinecone (corrected query format)
results = index.query(
    vector=query_embedding,  # Single query vector (list format)
    top_k=3,  # Number of top results you want to retrieve
    include_metadata=True,  # Include metadata in the results
    # namespace=namespace  # Use the namespace, if necessary
)

# Display the relevant documents (metadata)
print("\n--- Relevant Documents ---")
for i, match in enumerate(results['matches'], 1):
    print(f"Document {i}:\n{match['metadata']['text']}\n")
    if 'source' in match['metadata']:
        print(f"Source: {match['metadata'].get('source', 'Unknown')}\n")
