import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Set up the persistent directory for ChromaDB
# This directory will store the vector database and its metadata
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chromadb_books")

# Initialize the embedding model
# Using MiniLM-L6-v2 for its good balance of speed and accuracy
# This model converts text into 384-dimensional vectors
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the vector store from the persistent directory
# This will load the vector database and its metadata
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Initialize the Language Model using Groq
# temperature=0.7 provides a good balance between creativity and accuracy
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user","{input}"),
      ("system", """Answer the user's question based on the following context:
    
        {context}
    
    If you cannot answer the question based on the context, say "I don't have enough information to answer that question."
    Include relevant quotes from the context to support your answer.""")
])

# Create a document chain
# This chain will combine the retrieved docs with the prompt and send it  the LLM

document_chain = create_stuff_documents_chain(
    llm = llm,
    prompt = prompt,
)

# Create a retrieval chain 
# this change will handle complete flow

retrieval_chain = create_retrieval_chain(
    retriever = retriever,
    combine_docs_chain = document_chain,
)

def chat_loop():
    """
    Implements the main chat loop for the RAG chatbot.
    - Maintains chat history
    - Handles user input
    - Processes responses
    - Manages the conversation flow
    """
    chat_history: List = []
    print("Welcome to the RAG Chatbot! Type 'quit' to exit.")
    
    while True:
        # Get user input and handle exit commands
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        # Process the user input through the retrieval chain
        # Include chat history for context-aware responses
        response = retrieval_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        # Update the chat history with the new interaction
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response["answer"])
        ])
        
        # Display the assistant's response
        print("\nAssistant:", response["answer"])

# Entry point of the script
if __name__ == "__main__":
    chat_loop()