from langchain_community.document_loaders import PyPDFLoader
import os
from langchain.text_splitter import CharacterTextSplitter

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data", "The_Art_of_War.pdf")

loader = PyPDFLoader(file_path)
docs = loader.load()
# print(docs)
# print(docs[0].page_content)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)
# print(chunks)
# print(chunks[0].page_content)
# print chunks in a list with a separator
print("\n *********** \n".join([chunk.page_content for chunk in chunks]))
