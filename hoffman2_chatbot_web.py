# hoffman2_chatbot_web.py
# Description: Modified version of hoffman2_chatbot.py for web integration.
# This script contains the core logic for loading the LlamaIndex
# and querying it. CLI-specific parts have been removed or adapted.

import os
import sys
import time
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.openai import OpenAIEmbedding # Ensure this is the correct import
from llama_index.llms.openai import OpenAI # Ensure this is the correct import

# Global variable to store the query engine, initialized by Flask app
# query_engine = None # This will be managed by app.py

def load_environment():
    """
    Load environment variables and check for API key.
    This should be called once when the application starts.
    """
    load_dotenv() # Load variables from .env file
    if not os.getenv("OPENAI_API_KEY"):
        # In a web app, raising an error is better than sys.exit
        raise ValueError("OPENAI_API_KEY not found. Please create a .env file or set the environment variable.")

    # Configure LlamaIndex settings (models, etc.)
    # These settings are used globally by LlamaIndex
    try:
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.llm = OpenAI(model="gpt-4o") # or your preferred model
        print("LlamaIndex OpenAI models configured.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI models for LlamaIndex: {e}")


def load_hoffman2_index(index_path="./hoffman2_index"):
    """
    Load the Hoffman2 documentation index from disk.
    Returns the query_engine.
    """
    print(f"Attempting to load Hoffman2 documentation index from {index_path}...")

    if not os.path.exists(index_path) or not os.path.isdir(index_path):
        # In a web app, it's better to raise an error or return None
        # and let the calling function handle it.
        print(f"Error: Index directory not found at {index_path}.")
        print("Please ensure 'createdb.py' has been run successfully and the index is in the correct location.")
        raise FileNotFoundError(f"Index directory not found: {index_path}")

    try:
        # Load the storage context
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        # Load the index from storage
        index = load_index_from_storage(storage_context)

        # Create a query engine from the index
        # You can customize similarity_top_k, response_mode, etc.
        loaded_query_engine = index.as_query_engine(
            similarity_top_k=5, # Retrieve top 5 most similar nodes
            response_mode="compact" # Optimizes for shorter, more direct answers
        )
        print("Hoffman2 documentation index loaded successfully into query engine.")
        return loaded_query_engine
    except Exception as e:
        # Log the error and raise a more specific exception or return None
        print(f"Critical Error: Failed to load index from {index_path}. Exception: {e}")
        print("The index files might be corrupted, incompatible, or missing critical components.")
        print("Consider rebuilding the index by running 'createdb.py' again.")
        # Re-raise the exception so the Flask app can handle it, or raise a custom one
        raise RuntimeError(f"Failed to load LlamaIndex: {e}")


def ask_hoffman2_web(query_engine, question: str):
    """
    Function to query the Hoffman2 documentation index.
    This version is for the web app and takes the query_engine as an argument.
    """
    if not query_engine:
        # This should ideally be caught before calling this function
        raise ValueError("Query engine is not initialized.")
    if not question or not isinstance(question, str):
        raise ValueError("Question must be a non-empty string.")

    start_time = time.time()

    # Determine if the question is about workshops to tailor the prompt
    workshop_keywords = ["workshop", "course", "tutorial", "training", "class", "lesson", "seminar"]
    is_workshop_question = any(keyword in question.lower() for keyword in workshop_keywords)

    # Enhanced prompt for better context and user assistance
    if is_workshop_question:
        # Prompt specifically for workshop-related queries
        enhanced_question = f"""
        You are an AI assistant for UCLA's Hoffman2 HPC cluster.
        A user is asking about workshops or training. Please answer the following question:
        "{question}"

        Provide details about relevant workshops, including topics covered and how they might benefit the user.
        If specific workshop names or links are in your knowledge base, please include them.
        Your goal is to guide the user to appropriate learning resources.
        """
    else:
        # General prompt, emphasizing help for novice users
        enhanced_question = f"""
        You are an AI assistant for UCLA's Hoffman2 HPC cluster, designed to help users of all skill levels, especially novices.
        Please answer the following question clearly and concisely:
        "{question}"

        If the question seems basic or from a beginner, provide step-by-step instructions if applicable.
        Use clear language and avoid excessive jargon. If technical terms are necessary, briefly explain them.
        If the question involves commands, provide the exact syntax and an example if possible.
        Your primary goal is to be helpful and make Hoffman2 more accessible.
        """
    
    print(f"Querying index with enhanced question: {enhanced_question[:100]}...") # Log snippet of question
    
    try:
        response = query_engine.query(enhanced_question)
        answer = response.response # Extract the text part of the response
    except Exception as e:
        print(f"Error during LlamaIndex query: {e}")

        answer = "I apologize, but I encountered an error trying to find an answer. Please try rephrasing your question or try again later."

    end_time = time.time()
    query_duration = end_time - start_time
    
    print(f"Query completed in {query_duration:.2f} seconds.")
    return answer, query_duration

