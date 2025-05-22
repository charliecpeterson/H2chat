import os
import sys
import time
import argparse
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

def load_environment():
    """Load environment variables and check for API key"""
    # Load variables from .env file
    load_dotenv()
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please create a .env file with your API key.")
    
    # Create embedding and LLM instances
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-4o")
    
    # Set up settings globally
    Settings.embed_model = embed_model
    Settings.llm = llm

def load_hoffman2_index(index_path="./hoffman2_index"):
    """Load the Hoffman2 documentation index from disk"""
    print(f"Loading Hoffman2 documentation index from {index_path}...")
    
    # Check if the index exists
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}.")
        print("Please run createdb.py first to build the documentation index.")
        sys.exit(1)
    
    # Simple loading without any patching
    try:
        # Load the storage context
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        
        # Create a query engine
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )
        
        return query_engine
    except Exception as e:
        print(f"Error loading index: {e}")
        print("The index files may be corrupted or using an incompatible encoding.")
        print("Consider rebuilding the index by running createdb.py again.")
        sys.exit(1)

def ask_hoffman2(query_engine, question):
    """Function to query the Hoffman2 documentation index"""
    start_time = time.time()
    
    # Check if the question is specifically about workshops
    workshop_keywords = ["workshop", "course", "tutorial", "training", "class", "lesson"]
    is_workshop_question = any(keyword in question.lower() for keyword in workshop_keywords)
    
    # Improve the prompt with context about novice users
    if is_workshop_question:
        enhanced_question = f"""
        As an assistant helping users find Hoffman2 workshops and training materials, please answer this question:
        
        {question}
        
        If there are relevant workshops available, provide details about them and what topics they cover.
        Focus on practical information about what users can learn from these workshops.
        """
    else:
        enhanced_question = f"""
        As an assistant helping novice HPC users on the Hoffman2 cluster, please answer this question:
        
        {question}
        
        If this is a beginner question, provide step-by-step instructions with examples.
        Focus on practical information rather than technical jargon.
        If commands are needed, show the exact syntax.
        """
    
    response = query_engine.query(enhanced_question)
    end_time = time.time()
    return response.response, end_time - start_time

from llama_index.core.memory import ChatMemoryBuffer

def interactive_mode(query_engine):
    """Run an interactive chat session with the Hoffman2 documentation chatbot"""
    print("\n=== Hoffman2 Documentation Chatbot ===")
    print("Type your questions about Hoffman2 HPC cluster below.")
    print("Type 'help' to see common commands or topics.")
    print("Type 'exit', 'quit', or 'q' to end the session.\n")
    
    # Common commands and examples for novices
    help_topics = {
        "login": "How do I log in to Hoffman2?",
        "transfer": "How do I transfer files to Hoffman2?",
        "job": "How do I submit a job on Hoffman2?",
        "modules": "How do I use software modules on Hoffman2?",
        "storage": "How do I manage storage on Hoffman2?",
        "queue": "How do I check my job status on Hoffman2?",
        "compute": "What compute resources are available on Hoffman2?",
        "workshops": "What workshops are available for learning Hoffman2?",
        "ml": "Are there any machine learning workshops for Hoffman2?",
        "containers": "How do I use containers on Hoffman2?",
    }
    
    # Keep track of conversation context without using chat_engine
    conversation_context = []
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() in ["exit", "quit", "q"]:
            print("Thank you for using the Hoffman2 Documentation Chatbot!")
            break
        
        if not question.strip():
            continue
            
        if question.lower() == "help":
            print("\n=== Common Topics for Beginners ===")
            for cmd, example in help_topics.items():
                print(f"  {cmd.ljust(10)} - Try asking: '{example}'")
            print("\nYou can also ask about any other Hoffman2 topic.")
            continue
            
        if question.lower() in help_topics:
            # If user just types a command name, use the example question
            question = help_topics[question.lower()]
            print(f"Answering: {question}")
        
        # Add simple conversation context for follow-up questions
        if len(conversation_context) > 0 and not question.endswith("?"):
            # If this looks like a follow-up question, include context
            enhanced_question = f"""
            Previous question: {conversation_context[-1]}
            
            Follow-up question: {question}
            
            Please answer the follow-up question taking into account the previous context.
            """
        else:
            enhanced_question = question
            
        # Remember this question for context
        conversation_context.append(question)
        if len(conversation_context) > 3:
            # Keep only the most recent questions
            conversation_context = conversation_context[-3:]
            
        print("\nSearching documentation...")
        response, query_time = ask_hoffman2(query_engine, enhanced_question)
        print(f"Answer (found in {query_time:.2f} seconds):\n")
        print(response)
        print("\n" + "-"*50)

def tutorial_mode(query_engine):
    """Run a guided tutorial for new Hoffman2 users"""
    tutorial_steps = [
        "What is Hoffman2 and what can it be used for?",
        "How do I get access to Hoffman2?",
        "How do I connect to Hoffman2 for the first time?",
        "How do I transfer files to and from Hoffman2?",
        "How do I run a simple job on Hoffman2?",
        "How do I check the status of my jobs on Hoffman2?",
        "What software is available on Hoffman2 and how do I use it?",
        "How do I get help with Hoffman2 issues?"
    ]
    
    print("\n=== Hoffman2 Guided Tutorial for Beginners ===")
    print("This tutorial will walk you through the basics of using Hoffman2.")
    print("Press ENTER after each step to continue, or type 'quit' to exit.\n")
    
    for i, question in enumerate(tutorial_steps):
        print(f"\nStep {i+1}/{len(tutorial_steps)}: {question}")
        response, _ = ask_hoffman2(query_engine, question)
        print("\nAnswer:\n")
        print(response)
        print("\n" + "-"*50)
        
        user_input = input("\nPress ENTER to continue or type 'quit' to exit: ")
        if user_input.lower() == 'quit':
            print("Tutorial ended. You can restart any time.")
            return

 
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Hoffman2 Documentation Chatbot")
    parser.add_argument("--index-path", default="./hoffman2_index", help="Path to the index directory")
    parser.add_argument("--question", help="Single question mode: ask a question and exit")
    parser.add_argument("--tutorial", action="store_true", help="Run the guided tutorial for beginners")
    args = parser.parse_args()
    
    # Load environment variables
    load_environment()
    
    # Load the Hoffman2 documentation index
    query_engine = load_hoffman2_index(args.index_path)
    
    if args.tutorial:
        # Run the guided tutorial
        tutorial_mode(query_engine)
    elif args.question:
        # Single question mode
        print(f"Question: {args.question}")
        response, query_time = ask_hoffman2(query_engine, args.question)
        print(f"Answer (found in {query_time:.2f} seconds):")
        print(response)
    else:
        # Interactive mode
        interactive_mode(query_engine)

if __name__ == "__main__":
    main()