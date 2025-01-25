import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

print(pinecone_api_key)

# Initialize Pinecone with your API key
pc = Pinecone(api_key=pinecone_api_key)  # replace with your Pinecone API key

# Define Pinecone index name and namespace
index_name = "llm-training-facts"
namespace = ""

# Initialize the Pinecone index
index = pc.Index(index_name)

# Define the directory containing the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data", "facts.txt")

# Ensure the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

loader = TextLoader(file_path)
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")

# Create embeddings using HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Prepare the documents and embeddings for Pinecone
print("Creating Embeddings")
embedding_vectors = []

for idx, doc in enumerate(docs):
    # Generate embedding for each document chunk
    embedding = embeddings.embed_query(doc.page_content)
    
    # Create a unique ID for each document chunk
    doc_id = f"doc_{idx}"
    
    # Append the vector (embedding), metadata, and ID to the list
    embedding_vectors.append({
        "id": doc_id,
        "values": embedding,
        "metadata": {"text": doc.page_content}  # You can store additional metadata here
    })

# Upsert embeddings into Pinecone
index.upsert(vectors=embedding_vectors)

print(f"Embeddings for {len(docs)} document chunks have been upserted into Pinecone.")
