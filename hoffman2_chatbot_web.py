# hoffman2_chatbot_web.py

import os
import sys
import time
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.openai import OpenAIEmbedding # Ensure this is the correct import
from llama_index.llms.openai import OpenAI # Ensure this is the correct import


def load_environment():
    """
    Load environment variables and check for API key.
    This should be called once when the application starts.
    """
    load_dotenv() 
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Set the environment variable.")

    try:
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        Settings.llm = OpenAI(model="gpt-4o")
        print("LlamaIndex OpenAI models configured.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI models: {e}")


def load_hoffman2_index(index_path="./hoffman2_index"):
    """
    Load the Hoffman2 documentation index from disk.
    Returns the query_engine.
    """
    print(f"Attempting to load Hoffman2 documentation index from {index_path}...")

    if not os.path.exists(index_path) or not os.path.isdir(index_path):
        print(f"Error: Index directory not found at {index_path}.")
        print("Please ensure 'createdb.py' has been run successfully and the index is in the correct location.")
        raise FileNotFoundError(f"Index directory not found: {index_path}")

    try:
        # Load the storage context
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        # Load the index from storage
        index = load_index_from_storage(storage_context)

        # Create a query engine from the index
        loaded_query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )
        print("Hoffman2 documentation index loaded successfully into query engine.")
        return loaded_query_engine
    except Exception as e:
        print(f"Critical Error: Failed to load index from {index_path}. Exception: {e}")
        print("The index files might be corrupted, incompatible, or missing critical components.")
        print("Consider rebuilding the index by running 'createdb.py' again.")
        raise RuntimeError(f"Failed to load LlamaIndex: {e}")


def get_support_message():
    """
    Returns a formatted message with support ticket link.
    """
    return "\n\n---\n**Need additional help?** [Submit a support ticket](https://support.idre.ucla.edu/helpdesk/Tickets/New) for personalized assistance from the IDRE team."


def ask_hoffman2_web(query_engine, question: str, chat_history: list = None):
    """
    Function to query the Hoffman2 documentation index, considering chat history.
    """
    if not query_engine:
        raise ValueError("Query engine is not initialized.")
    if not question or not isinstance(question, str):
        raise ValueError("Question must be a non-empty string.")

    start_time = time.time()

    # Construct a history string to prepend to the prompt
    history_context = ""
    if chat_history:
        for entry in chat_history:
            # Ensure roles are mapped correctly if frontend uses different terms
            role = "User" if entry.get('role') == 'user' else "Assistant"
            history_context += f"{role}: {entry.get('content', '')}\n" # Use .get for safety
        history_context += "\n---\n" # Separator for clarity

    enhanced_question = f"""
You are an AI assistant for UCLA's Hoffman2 HPC cluster.
Your knowledge comes from the Hoffman2 documentation.
Refer to the following conversation history for context if it's relevant to the current question.
If the history is not relevant, or if the current question is a new topic, answer the current question directly.

Conversation History:
{history_context if chat_history else "No previous conversation history."}

Current User Question:
User: "{question}"

Provide clear, concise, and accurate information based *only* on the Hoffman2 documentation you have been trained on.
If the question involves commands, provide the exact syntax and examples when helpful.
If the answer is not in the documentation, state that the information is not available in your current knowledge base and suggest submitting a support ticket at https://support.idre.ucla.edu/helpdesk/Tickets/New for personalized assistance.
Do not invent information or answer questions outside the scope of Hoffman2.
Format your response using Markdown. For example, use backticks for inline code, and triple backticks for code blocks.

A:
"""

    print(f"Querying index with enhanced question (includes history if present). First 200 chars of prompt: {enhanced_question[:200]}...")

    try:
        response = query_engine.query(enhanced_question)
        answer = response.response
        
        # Add support link if the response indicates uncertainty
        if any(phrase in answer.lower() for phrase in ["not available", "don't know", "not sure", "cannot find"]):
            answer += get_support_message()
            
    except Exception as e:
        print(f"Error during LlamaIndex query: {e}")
        answer = f"I apologize, but I encountered an error trying to find an answer. Please try rephrasing your question.{get_support_message()}"

    end_time = time.time()
    query_duration = end_time - start_time

    print(f"Query completed in {query_duration:.2f} seconds.")
    return answer, query_duration
