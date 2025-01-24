import os

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
# Define the directory containing the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "war_and_peace.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

loader = TextLoader(file_path)
documents = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")
        
# 1. Character Text Splitter
# Splits text into chunks based on the number of characters
# USeful for consistent chunk size regardless of content structure
print("_______Using Character Splitter_______")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)
#print the docs
# print(docs[0].page_content, docs[1].page_content, docs[2].page_content)
print("--------------Chunk 1------------------")
print(docs[0].page_content)
print("--------------Chunk 2------------------")
print(docs[1].page_content)
print("--------------Chunk 3------------------")
print(docs[2].page_content)
create_vector_store(docs, "chroma_db_char")


# 2. Sentenced based text splitter
# Splits text into chunks based on the number of sentences
# Useful for consistent chunk size regardless of content structure
print("_______Using Sentenced based Splitter_______")
text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)
#print the docs
print(docs[0].page_content, docs[1].page_content, docs[2].page_content)
print("--------------Chunk 1------------------")
print(docs[0].page_content)
print("--------------Chunk 2------------------")
print(docs[1].page_content)
print("--------------Chunk 3------------------")
print(docs[2].page_content)
create_vector_store(docs, "chroma_db_sentenced")

# 3. Token based text splitter
# Splits text into chunks based on the number of tokens
# Useful for consistent chunk size regardless of content structure
print("_______Using Token based Splitter_______")
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)
#print the docs
print(docs[0].page_content, docs[1].page_content, docs[2].page_content)
print("--------------Chunk 1------------------")
print(docs[0].page_content)
print("--------------Chunk 2------------------")
print(docs[1].page_content)
print("--------------Chunk 3------------------")
print(docs[2].page_content)
create_vector_store(docs, "chroma_db_token")

# 4. Recursive Character-based Splitting
# Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
# Balances between maintaining coherence and adhering to character limits.
print("\n--- Using Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, "chroma_db_rec_char")


# 5. Custom Text Splitter

class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        # Custom logic for splitting text
        return text.split("\n\n")  # Example: split by paragraphs

custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")